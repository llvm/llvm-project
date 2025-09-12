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
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

#include <atomic>
#include <future>
#include <mutex>
#include <unordered_map>

#include <cassert>
#include <utility>

using namespace clang;

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

Value::Value(const Value &RHS) : Ty(RHS.getType()), VKind(K_None) {
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
    for (uint64_t I = 0, N = RHS.getArraySize(); I < N; ++I)
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
    MakeStr(RHS.getStrVal().str());
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

public:
  ValueReadVisitor(ValueReaderDispatcher &D, llvm::orc::ExecutorAddr A)
      : Dispatcher(D), Addr(A) {}

  llvm::Expected<Value> VisitType(const Type *T) {
    return Dispatcher.readOtherObject(QualType(T, 0), Addr);
  }

  llvm::Expected<Value> VisitBuiltinType(const BuiltinType *BT) {
    return Dispatcher.readBuiltin(QualType(BT, 0), Addr);
  }

  llvm::Expected<Value> VisitPointerType(const PointerType *PT) {
    return Dispatcher.readPointer(QualType(PT, 0), Addr);
  }

  llvm::Expected<Value> VisitConstantArrayType(const ConstantArrayType *AT) {
    return Dispatcher.readArray(QualType(AT, 0), Addr);
  }

  llvm::Expected<Value> VisitEnumType(const EnumType *ET) {
    return Dispatcher.readBuiltin(QualType(ET, 0), Addr);
  }
};

llvm::Expected<Value>
ValueReaderDispatcher::read(QualType QT, llvm::orc::ExecutorAddr Addr) {
  ValueReadVisitor V(*this, Addr);
  return V.Visit(QT.getTypePtr());
}

llvm::Expected<Value>
ValueReaderDispatcher::readBuiltin(QualType Ty, llvm::orc::ExecutorAddr Addr) {
  if (Ty->isVoidType())
    return Value();
  auto Size = Ctx.getTypeSizeInChars(Ty).getQuantity();
  auto ResOrErr = MA.readBuffers({llvm::orc::ExecutorAddrRange(Addr, Size)});
  if (!ResOrErr)
    return ResOrErr.takeError();

  const auto &Res = *ResOrErr;
  return Value(Ty, Res.back());
}

llvm::Expected<Value>
ValueReaderDispatcher::readPointer(QualType Ty, llvm::orc::ExecutorAddr Addr) {
  auto PtrTy = dyn_cast<PointerType>(Ty.getTypePtr());
  if (!PtrTy)
    return llvm::make_error<llvm::StringError>("Not a PointerType",
                                               llvm::inconvertibleErrorCode());

  uint64_t PtrValAddr = Addr.getValue();
  if (PtrValAddr == 0)
    return Value(Ty, PtrValAddr); // null pointer

  llvm::orc::ExecutorAddr PointeeAddr(PtrValAddr);
  Value Val(Ty, PtrValAddr);

  QualType PointeeTy = PtrTy->getPointeeType();
  if (PointeeTy->isCharType()) {
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
    Value Str(PointeeTy, S.c_str());
    Val.getPointerPointee() = std::move(Str);
  }

  return std::move(Val);
}

llvm::Expected<Value>
ValueReaderDispatcher::readArray(QualType Ty, llvm::orc::ExecutorAddr Addr) {
  const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(Ty);
  if (!CAT)
    return llvm::make_error<llvm::StringError>("Not a ConstantArrayType",
                                               llvm::inconvertibleErrorCode());

  QualType ElemTy = CAT->getElementType();
  size_t ElemSize = Ctx.getTypeSizeInChars(ElemTy).getQuantity();

  Value Val(Value::UninitArr(), Ty, CAT->getZExtSize());
  for (size_t i = 0; i < CAT->getZExtSize(); ++i) {
    auto ElemAddr = Addr + i * ElemSize;
    if (ElemTy->isPointerType()) {
      auto BufOrErr = MA.readUInt64s({ElemAddr});
      if (!BufOrErr)
        return BufOrErr.takeError();
      llvm::orc::ExecutorAddr Addr(BufOrErr->back());
      ElemAddr = Addr;
    }

    auto ElemBufOrErr = read(ElemTy, ElemAddr);
    if (!ElemBufOrErr)
      return ElemBufOrErr.takeError();
    Val.getArrayInitializedElt(i) = std::move(*ElemBufOrErr);
  }

  return std::move(Val);
}

llvm::Expected<Value>
ValueReaderDispatcher::readOtherObject(QualType Ty,
                                       llvm::orc::ExecutorAddr Addr) {
  llvm::outs() << Addr.getValue();
  if (Ty->isRecordType()) {
    llvm::outs() << "Here in recordtype\n";
    auto BufOrErr = MA.readUInt64s({Addr});
    if (!BufOrErr)
      return BufOrErr.takeError();
    Addr = llvm::orc::ExecutorAddr(BufOrErr->back());
  }
  uint64_t PtrValAddr = Addr.getValue();
  llvm::outs() << PtrValAddr;
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
  QualType Ty;

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = IdToType.find(ID);
    if (It == IdToType.end()) {
      SendResult(llvm::make_error<llvm::StringError>(
          "Unknown ValueId in deliverResult", llvm::inconvertibleErrorCode()));
    }
    Ty = It->second.getCanonicalType();
    IdToType.erase(It);
  }

  ValueReaderDispatcher Runner(Ctx, MemAcc);
  auto BufOrErr = Runner.read(Ty, Addr);

  if (!BufOrErr) {
    SendResult(BufOrErr.takeError());
    return;
  }

  // Store the successfully read value buffer
  LastVal = std::move(*BufOrErr);
  SendResult(llvm::Error::success());
  return;
}
} // namespace clang
