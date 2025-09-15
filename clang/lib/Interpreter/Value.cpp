//===------------ Value.cpp - Definition of interpreter value -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the class that used to represent a value in incremental
// C++.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"

#include "InterpreterUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

#include <atomic>
#include <future>
#include <mutex>
#include <unordered_map>

#include <cassert>
#include <utility>

using namespace clang;

#define DEBUG_TYPE "value"

namespace {

// This is internal buffer maintained by Value, used to hold temporaries.
class ValueStorage {
public:
  using DtorFunc = void (*)(void *);

  static unsigned char *CreatePayload(void *DtorF, size_t AllocSize,
                                      size_t ElementsSize) {
    if (AllocSize < sizeof(Canary))
      AllocSize = sizeof(Canary);
    unsigned char *Buf =
        new unsigned char[ValueStorage::getPayloadOffset() + AllocSize];
    ValueStorage *VS = new (Buf) ValueStorage(DtorF, AllocSize, ElementsSize);
    std::memcpy(VS->getPayload(), Canary, sizeof(Canary));
    return VS->getPayload();
  }

  unsigned char *getPayload() { return Storage; }
  const unsigned char *getPayload() const { return Storage; }

  static unsigned getPayloadOffset() {
    static ValueStorage Dummy(nullptr, 0, 0);
    return Dummy.getPayload() - reinterpret_cast<unsigned char *>(&Dummy);
  }

  static ValueStorage *getFromPayload(void *Payload) {
    ValueStorage *R = reinterpret_cast<ValueStorage *>(
        (unsigned char *)Payload - getPayloadOffset());
    return R;
  }

  void Retain() { ++RefCnt; }

  void Release() {
    assert(RefCnt > 0 && "Can't release if reference count is already zero");
    if (--RefCnt == 0) {
      // We have a non-trivial dtor.
      if (Dtor && IsAlive()) {
        assert(Elements && "We at least should have 1 element in Value");
        size_t Stride = AllocSize / Elements;
        for (size_t Idx = 0; Idx < Elements; ++Idx)
          (*Dtor)(getPayload() + Idx * Stride);
      }
      delete[] reinterpret_cast<unsigned char *>(this);
    }
  }

  // Check whether the storage is valid by validating the canary bits.
  // If someone accidentally write some invalid bits in the storage, the canary
  // will be changed first, and `IsAlive` will return false then.
  bool IsAlive() const {
    return std::memcmp(getPayload(), Canary, sizeof(Canary)) != 0;
  }

private:
  ValueStorage(void *DtorF, size_t AllocSize, size_t ElementsNum)
      : RefCnt(1), Dtor(reinterpret_cast<DtorFunc>(DtorF)),
        AllocSize(AllocSize), Elements(ElementsNum) {}

  mutable unsigned RefCnt;
  DtorFunc Dtor = nullptr;
  size_t AllocSize = 0;
  size_t Elements = 0;
  unsigned char Storage[1];

  // These are some canary bits that are used for protecting the storage been
  // damaged.
  static constexpr unsigned char Canary[8] = {0x4c, 0x37, 0xad, 0x8f,
                                              0x2d, 0x23, 0x95, 0x91};
};
} // namespace

namespace clang {

// static Value::Kind ConvertQualTypeToKind(const ASTContext &Ctx, QualType QT)
// {
//   if (Ctx.hasSameType(QT, Ctx.VoidTy))
//     return Value::K_Void;

//   if (const auto *ED = QT->getAsEnumDecl())
//     QT = ED->getIntegerType();

//   const auto *BT = QT->getAs<BuiltinType>();
//   if (!BT || BT->isNullPtrType())
//     return Value::K_PtrOrObj;

//   switch (QT->castAs<BuiltinType>()->getKind()) {
//   default:
//     assert(false && "Type not supported");
//     return Value::K_Unspecified;
// #define X(type, name) \
//   case BuiltinType::name: \
//     return Value::K_##name;
//     REPL_BUILTIN_TYPES
// #undef X
//   }
// }

Value::Value(const Value &RHS)
    : Ty(RHS.getType()), VKind(K_None), Cleanup(RHS.Cleanup) {
  switch (RHS.getKind()) {
  case K_None:
    VKind = RHS.VKind;
    break;
  case K_Builtin: {
    MakeBuiltIns();
    if (RHS.asBuiltin().getKind() != BuiltinKind::K_Unspecified) {
      setBuiltins(asBuiltin(), RHS.asBuiltin());
    }
    break;
  }
  case K_Array: {
    MakeArray(RHS.getArraySize());
    for (uint64_t I = 0, N = RHS.getArrayInitializedElts(); I < N; ++I)
      getArrayInitializedElt(I) = RHS.getArrayInitializedElt(I);
    break;
  }
  case K_Pointer: {
    MakePointer(RHS.getAddr());
    if (RHS.HasPointee())
      getPointerPointee() = RHS.getPointerPointee();
    break;
  }
  case K_Str:
    MakeStr(RHS.getStrVal());
    break;
  }
}

Value &Value::operator=(const Value &RHS) {
  if (this != &RHS)
    *this = Value(RHS);

  return *this;
}

Value &Value::operator=(Value &&RHS) {
  if (this != &RHS) {
    if (VKind != K_None)
      destroy();

    Ty = RHS.Ty;
    VKind = RHS.VKind;
    Data = RHS.Data;
    Cleanup = std::move(RHS.Cleanup);
    RHS.VKind = K_None;
  }

  return *this;
}

Value::Value(QualType QT, std::vector<uint8_t> Raw) : Ty(QT), VKind(K_None) {
  MakeBuiltIns();
  Builtins &B = asBuiltin();
  if (const auto *ED = QT->getAsEnumDecl())
    QT = ED->getIntegerType();
  switch (QT->castAs<BuiltinType>()->getKind()) {
  default:
    assert(false && "Type not supported");

#define X(type, name)                                                          \
  case BuiltinType::name: {                                                    \
    B.setKind(BuiltinKind::K_##name);                                          \
    B.set##name(as<type>(Raw));                                                \
  } break;
    REPL_BUILTIN_TYPES
#undef X
  }
}

void Value::dump(ASTContext &Ctx) const { print(llvm::outs(), Ctx); }

void Value::printType(llvm::raw_ostream &Out, ASTContext &Ctx) const {
  Out << ValueToString(Ctx).toString(getType());
}

void Value::printData(llvm::raw_ostream &Out, ASTContext &Ctx) const {
  Out << ValueToString(Ctx).toString(this);
}
// FIXME: We do not support the multiple inheritance case where one of the base
// classes has a pretty-printer and the other does not.
void Value::print(llvm::raw_ostream &Out, ASTContext &Ctx) const {
  // Don't even try to print a void or an invalid type, it doesn't make sense.
  if (getType()->isVoidType() || isAbsent())
    return;

  // We need to get all the results together then print it, since `printType` is
  // much faster than `printData`.
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  SS << "(";
  printType(SS, Ctx);
  SS << ") ";
  printData(SS, Ctx);
  SS << "\n";
  Out << Str;
}

void ValueCleanup::operator()(Value &V) {
  using namespace llvm;

  if (!V.isPointer() || V.getAddr() != 0)
    return;

  LLVM_DEBUG(dbgs() << "ValueCleanup: destroying value at Addr=" << V.getAddr()
                    << ", Type=" << V.getType().getAsString() << "\n");
  assert(!DtorWrapperFn.isNull() &&
         "Expected valid destructor wrapper function address, but found null");
  if (ObjDtor) {
    auto ObjDtorAddrOrErr = ObjDtor(V.getType());
    if (ObjDtorAddrOrErr && !ObjDtorAddrOrErr->isNull()) {
      Error E = Error::success();
      orc::ExecutorAddr ObjDtorFn = *ObjDtorAddrOrErr;
      orc::ExecutorAddr Addr(V.getAddr());
      LLVM_DEBUG(dbgs() << "ValueCleanup: calling object-specific destructor, "
                           "Addr="
                        << Addr.getValue() << "\n");
      cantFail(ES->callSPSWrapper<orc::shared::SPSError(
                   orc::shared::SPSExecutorAddr, orc::shared::SPSExecutorAddr)>(
          DtorWrapperFn, E, ObjDtorFn, Addr));
      cantFail(std::move(E));
    } else {
      LLVM_DEBUG(dbgs() << "ValueCleanup: failed to get ObjDtor address\n");
      consumeError(ObjDtorAddrOrErr.takeError());
    }
  }

  assert(!DtorFn.isNull() &&
         "Expected valid destructor function address, but found null");

  Error E = Error::success();
  orc::ExecutorAddr Addr(V.getAddr());
  LLVM_DEBUG(dbgs() << "ValueCleanup: calling raw destructor, Addr="
                    << Addr.getValue() << "\n");
  cantFail(ES->callSPSWrapper<orc::shared::SPSError(
               orc::shared::SPSExecutorAddr, orc::shared::SPSExecutorAddr)>(
      DtorWrapperFn, E, DtorFn, Addr));
  cantFail(std::move(E));
  LLVM_DEBUG(dbgs() << "ValueCleanup: finished destruction for Addr="
                    << V.getAddr() << "\n");
}

class ValueReaderDispatcher {
private:
  ASTContext &Ctx;
  llvm::orc::MemoryAccess &MA;

public:
  ValueReaderDispatcher(ASTContext &Ctx, llvm::orc::MemoryAccess &MA)
      : Ctx(Ctx), MA(MA) {}

  llvm::Expected<Value> read(QualType QT, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<Value> readBuiltin(QualType Ty, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<Value> readPointer(QualType Ty, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<Value> readArray(QualType Ty, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<Value> readOtherObject(QualType Ty,
                                        llvm::orc::ExecutorAddr Addr);
  // TODO: record, function, etc.
};

class ValueReadVisitor
    : public TypeVisitor<ValueReadVisitor, llvm::Expected<Value>> {
  ValueReaderDispatcher &Dispatcher;
  llvm::orc::ExecutorAddr Addr;
  QualType QT;

public:
  ValueReadVisitor(ValueReaderDispatcher &D, llvm::orc::ExecutorAddr A,
                   QualType QT)
      : Dispatcher(D), Addr(A), QT(QT) {}

  llvm::Expected<Value> VisitType(const Type *T) {
    return Dispatcher.readOtherObject(QT, Addr);
  }

  llvm::Expected<Value> VisitBuiltinType(const BuiltinType *BT) {
    return Dispatcher.readBuiltin(QT, Addr);
  }

  llvm::Expected<Value> VisitPointerType(const PointerType *PT) {
    return Dispatcher.readPointer(QT, Addr);
  }

  llvm::Expected<Value> VisitConstantArrayType(const ConstantArrayType *AT) {
    return Dispatcher.readArray(QT, Addr);
  }

  llvm::Expected<Value> VisitEnumType(const EnumType *ET) {
    return Dispatcher.readBuiltin(QT, Addr);
  }
};

llvm::Expected<Value>
ValueReaderDispatcher::read(QualType QT, llvm::orc::ExecutorAddr Addr) {
  ValueReadVisitor V(*this, Addr, QT);
  return V.Visit(QT.getCanonicalType().getTypePtr());
}

llvm::Expected<Value>
ValueReaderDispatcher::readBuiltin(QualType QT, llvm::orc::ExecutorAddr Addr) {
  QualType Ty = QT.getCanonicalType();
  LLVM_DEBUG(llvm::dbgs() << "readBuiltin: start, Addr=" << Addr.getValue()
                          << ", Type=" << Ty.getAsString() << "\n");
  if (Ty->isVoidType()) {
    LLVM_DEBUG(llvm::dbgs()
               << "readBuiltin: void type, returning empty Value\n");
    return Value(Ty);
  }

  auto Size = Ctx.getTypeSizeInChars(Ty).getQuantity();
  auto ResOrErr = MA.readBuffers({llvm::orc::ExecutorAddrRange(Addr, Size)});
  if (!ResOrErr) {
    LLVM_DEBUG(llvm::dbgs() << "readBuiltin: failed to read memory\n");

    return ResOrErr.takeError();
  }

  const auto &Res = *ResOrErr;
  LLVM_DEBUG(llvm::dbgs() << "readBuiltin: read succeeded, last byte addr="
                          << Res.back().data() << "\n");
  return Value(QT, Res.back());
}

llvm::Expected<Value>
ValueReaderDispatcher::readPointer(QualType QT, llvm::orc::ExecutorAddr Addr) {
  QualType Ty = QT.getCanonicalType();
  LLVM_DEBUG(llvm::dbgs() << "readPointer: start, Addr=" << Addr.getValue()
                          << "\n");

  if (!Ty->isPointerType()) {
    LLVM_DEBUG(llvm::dbgs() << "readPointer: Not a PointerType!\n");

    return llvm::make_error<llvm::StringError>("Not a PointerType",
                                               llvm::inconvertibleErrorCode());
  }

  uint64_t PtrValAddr = Addr.getValue();
  LLVM_DEBUG(llvm::dbgs() << "readPointer: raw pointer value=0x"
                          << llvm::format_hex(PtrValAddr, 10) << "\n");

  if (PtrValAddr == 0) {
    LLVM_DEBUG(llvm::dbgs() << "readPointer: null pointer detected\n");
    return Value(QT, PtrValAddr); // null pointer
  }

  llvm::orc::ExecutorAddr PointeeAddr(PtrValAddr);
  Value Val(QT, PtrValAddr);

  QualType PointeeTy = Ty->getPointeeType();
  LLVM_DEBUG(llvm::dbgs() << "readPointer: pointee type="
                          << PointeeTy.getAsString() << "\n");
  if (PointeeTy->isCharType()) {
    LLVM_DEBUG(llvm::dbgs() << "readPointer: reading C-string at pointee addr="
                            << PointeeAddr.getValue() << "\n");
    std::string S;
    for (size_t i = 0; i < 1024; ++i) {
      auto CRes = MA.readUInt8s({PointeeAddr + i});
      if (!CRes)
        return CRes.takeError();
      char c = static_cast<char>(CRes->back());
      if (c == '\0')
        break;
      S.push_back(c);
    }

    LLVM_DEBUG(llvm::dbgs() << "readPointer: read string=\"" << S << "\"\n");

    Value Str(PointeeTy, S.c_str());
    Val.getPointerPointee() = std::move(Str);
  }

  LLVM_DEBUG(llvm::dbgs() << "readPointer: finished\n");
  return std::move(Val);
}

llvm::Expected<Value>
ValueReaderDispatcher::readArray(QualType QT, llvm::orc::ExecutorAddr Addr) {
  LLVM_DEBUG(llvm::dbgs() << "readArray: start, Addr=" << Addr.getValue()
                          << "\n");
  QualType Ty = QT.getCanonicalType();
  const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(Ty);
  if (!CAT) {
    LLVM_DEBUG(llvm::dbgs() << "readArray: Not a ConstantArrayType!\n");

    return llvm::make_error<llvm::StringError>("Not a ConstantArrayType",
                                               llvm::inconvertibleErrorCode());
  }
  QualType ElemTy = CAT->getElementType();
  size_t ElemSize = Ctx.getTypeSizeInChars(ElemTy).getQuantity();
  uint64_t NumElts = CAT->getZExtSize();

  LLVM_DEBUG(llvm::dbgs() << "readArray: element type size=" << ElemSize
                          << ", element count=" << NumElts << "\n");

  Value Val(Value::UninitArr(), QT, NumElts);
  for (size_t i = 0; i < NumElts; ++i) {
    auto ElemAddr = Addr + i * ElemSize;
    LLVM_DEBUG(llvm::dbgs() << "readArray: reading element[" << i
                            << "] at Addr=" << ElemAddr.getValue() << "\n");

    if (ElemTy->isPointerType()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "readArray: element[" << i << "] is pointer type\n");
      auto BufOrErr = MA.readUInt64s({ElemAddr});
      if (!BufOrErr)
        return BufOrErr.takeError();
      llvm::orc::ExecutorAddr PointeeAddr(BufOrErr->back());
      LLVM_DEBUG(llvm::dbgs()
                 << "readArray: pointer element[" << i
                 << "] points to Addr=" << PointeeAddr.getValue() << "\n");
      ElemAddr = PointeeAddr;
    }

    auto ElemBufOrErr = read(ElemTy, ElemAddr);
    if (!ElemBufOrErr)
      return ElemBufOrErr.takeError();
    Val.getArrayInitializedElt(i) = std::move(*ElemBufOrErr);
    LLVM_DEBUG(llvm::dbgs()
               << "readArray: element[" << i << "] successfully read\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "readArray: finished reading array\n");
  return std::move(Val);
}

llvm::Expected<Value>
ValueReaderDispatcher::readOtherObject(QualType Ty,
                                       llvm::orc::ExecutorAddr Addr) {
  uint64_t PtrValAddr = Addr.getValue();
  return Value(Ty, PtrValAddr);
}

ValueResultManager::ValueResultManager(ASTContext &Ctx,
                                       llvm::orc::MemoryAccess &MA)
    : Ctx(Ctx), MemAcc(MA) {}

std::unique_ptr<ValueResultManager>
ValueResultManager::Create(llvm::orc::LLJIT &EE, ASTContext &Ctx, bool IsOop) {
  auto &ES = EE.getExecutionSession();
  auto &EPC = ES.getExecutorProcessControl();
  auto VRMgr = std::make_unique<ValueResultManager>(Ctx, EPC.getMemoryAccess());
  if (IsOop)
    VRMgr->Initialize(EE);
  return VRMgr;
}

void ValueResultManager::Initialize(llvm::orc::LLJIT &EE) {
  auto &ES = EE.getExecutionSession();

  llvm::orc::ExecutionSession::JITDispatchHandlerAssociationMap Handlers;
  using OrcSendResultFn =
      llvm::orc::shared::SPSError(uint64_t, llvm::orc::shared::SPSExecutorAddr);

  const char *SendValFnTag = "___orc_rt_SendResultValue_tag";
#ifndef __APPLE__
  ++SendValFnTag;
#endif
  Handlers[ES.intern(SendValFnTag)] = ES.wrapAsyncWithSPS<OrcSendResultFn>(
      this, &ValueResultManager::deliverResult);

  llvm::cantFail(ES.registerJITDispatchHandlers(*EE.getPlatformJITDylib(),
                                                std::move(Handlers)));
}

void ValueResultManager::deliverResult(SendResultFn SendResult, ValueId ID,
                                       llvm::orc::ExecutorAddr Addr) {
  LLVM_DEBUG(llvm::dbgs() << "deliverResult called with ID=" << ID
                          << ", Addr=" << Addr.getValue() << "\n");
  QualType Ty;
  std::optional<ValueCleanup> VCOpt = std::nullopt;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = IdToType.find(ID);
    if (It == IdToType.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unknown ValueId=" << ID << " in deliverResult\n");
      SendResult(llvm::make_error<llvm::StringError>(
          "Unknown ValueId in deliverResult", llvm::inconvertibleErrorCode()));
    }
    Ty = It->second;
    LLVM_DEBUG(llvm::dbgs() << "Resolved Type for ID=" << ID << "\n");

    IdToType.erase(It);

    auto valIt = IdToValCleanup.find(ID);
    if (valIt != IdToValCleanup.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Found ValueCleanup for ID=" << ID << "\n");

      VCOpt.emplace(std::move(valIt->second));
      IdToValCleanup.erase(valIt);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No ValueCleanup for ID=" << ID << "\n");
    }
  }

  ValueReaderDispatcher Dispatcher(Ctx, MemAcc);
  auto BufOrErr = Dispatcher.read(Ty, Addr);

  if (!BufOrErr) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to read value for ID=" << ID << "\n");

    SendResult(BufOrErr.takeError());
    return;
  }

  // Store the successfully read value buffer
  LastVal = std::move(*BufOrErr);
  LLVM_DEBUG(llvm::dbgs() << "Successfully read value for ID=" << ID << "\n");

  if (VCOpt) {
    LLVM_DEBUG(llvm::dbgs() << "Attaching ValueCleanup for ID=" << ID << "\n");
    LastVal.setValueCleanup(std::move(*VCOpt));
  }
  SendResult(llvm::Error::success());
  return;
}
} // namespace clang
