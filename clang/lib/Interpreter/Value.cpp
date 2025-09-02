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

static Value::Kind ConvertQualTypeToKind(const ASTContext &Ctx, QualType QT) {
  if (Ctx.hasSameType(QT, Ctx.VoidTy))
    return Value::K_Void;

  if (const auto *ED = QT->getAsEnumDecl())
    QT = ED->getIntegerType();

  const auto *BT = QT->getAs<BuiltinType>();
  if (!BT || BT->isNullPtrType())
    return Value::K_PtrOrObj;

  switch (QT->castAs<BuiltinType>()->getKind()) {
  default:
    assert(false && "Type not supported");
    return Value::K_Unspecified;
#define X(type, name)                                                          \
  case BuiltinType::name:                                                      \
    return Value::K_##name;
    REPL_BUILTIN_TYPES
#undef X
  }
}

Value::Value(const Interpreter *In, void *Ty) : Interp(In), OpaqueType(Ty) {
  const ASTContext &C = getASTContext();
  setKind(ConvertQualTypeToKind(C, getType()));
  if (ValueKind == K_PtrOrObj) {
    QualType Canon = getType().getCanonicalType();
    if ((Canon->isPointerType() || Canon->isObjectType() ||
         Canon->isReferenceType()) &&
        (Canon->isRecordType() || Canon->isConstantArrayType() ||
         Canon->isMemberPointerType())) {
      IsManuallyAlloc = true;
      // Compile dtor function.
      const Interpreter &Interp = getInterpreter();
      void *DtorF = nullptr;
      size_t ElementsSize = 1;
      QualType DtorTy = getType();

      if (const auto *ArrTy =
              llvm::dyn_cast<ConstantArrayType>(DtorTy.getTypePtr())) {
        DtorTy = ArrTy->getElementType();
        llvm::APInt ArrSize(sizeof(size_t) * 8, 1);
        do {
          ArrSize *= ArrTy->getSize();
          ArrTy = llvm::dyn_cast<ConstantArrayType>(
              ArrTy->getElementType().getTypePtr());
        } while (ArrTy);
        ElementsSize = static_cast<size_t>(ArrSize.getZExtValue());
      }
      if (auto *CXXRD = DtorTy->getAsCXXRecordDecl()) {
        if (llvm::Expected<llvm::orc::ExecutorAddr> Addr =
                Interp.CompileDtorCall(CXXRD))
          DtorF = reinterpret_cast<void *>(Addr->getValue());
        else
          llvm::logAllUnhandledErrors(Addr.takeError(), llvm::errs());
      }

      size_t AllocSize =
          getASTContext().getTypeSizeInChars(getType()).getQuantity();
      unsigned char *Payload =
          ValueStorage::CreatePayload(DtorF, AllocSize, ElementsSize);
      setPtr((void *)Payload);
    }
  }
}

Value::Value(const Value &RHS)
    : Interp(RHS.Interp), OpaqueType(RHS.OpaqueType), Data(RHS.Data),
      ValueKind(RHS.ValueKind), IsManuallyAlloc(RHS.IsManuallyAlloc) {
  if (IsManuallyAlloc)
    ValueStorage::getFromPayload(getPtr())->Retain();
}

Value::Value(Value &&RHS) noexcept {
  Interp = std::exchange(RHS.Interp, nullptr);
  OpaqueType = std::exchange(RHS.OpaqueType, nullptr);
  Data = RHS.Data;
  ValueKind = std::exchange(RHS.ValueKind, K_Unspecified);
  IsManuallyAlloc = std::exchange(RHS.IsManuallyAlloc, false);

  if (IsManuallyAlloc)
    ValueStorage::getFromPayload(getPtr())->Release();
}

Value &Value::operator=(const Value &RHS) {
  if (IsManuallyAlloc)
    ValueStorage::getFromPayload(getPtr())->Release();

  Interp = RHS.Interp;
  OpaqueType = RHS.OpaqueType;
  Data = RHS.Data;
  ValueKind = RHS.ValueKind;
  IsManuallyAlloc = RHS.IsManuallyAlloc;

  if (IsManuallyAlloc)
    ValueStorage::getFromPayload(getPtr())->Retain();

  return *this;
}

Value &Value::operator=(Value &&RHS) noexcept {
  if (this != &RHS) {
    if (IsManuallyAlloc)
      ValueStorage::getFromPayload(getPtr())->Release();

    Interp = std::exchange(RHS.Interp, nullptr);
    OpaqueType = std::exchange(RHS.OpaqueType, nullptr);
    ValueKind = std::exchange(RHS.ValueKind, K_Unspecified);
    IsManuallyAlloc = std::exchange(RHS.IsManuallyAlloc, false);

    Data = RHS.Data;
  }
  return *this;
}

void Value::clear() {
  if (IsManuallyAlloc)
    ValueStorage::getFromPayload(getPtr())->Release();
  ValueKind = K_Unspecified;
  OpaqueType = nullptr;
  Interp = nullptr;
  IsManuallyAlloc = false;
}

Value::~Value() { clear(); }

void *Value::getPtr() const {
  assert(ValueKind == K_PtrOrObj);
  return Data.m_Ptr;
}

void Value::setRawBits(void *Ptr, unsigned NBits /*= sizeof(Storage)*/) {
  assert(NBits <= sizeof(Storage) && "Greater than the total size");
  memcpy(/*dest=*/Data.m_RawBits, /*src=*/Ptr, /*nbytes=*/NBits / 8);
}

QualType Value::getType() const {
  return QualType::getFromOpaquePtr(OpaqueType);
}

const Interpreter &Value::getInterpreter() const {
  assert(Interp != nullptr &&
         "Can't get interpreter from a default constructed value");
  return *Interp;
}

const ASTContext &Value::getASTContext() const {
  return getInterpreter().getASTContext();
}

void Value::dump() const { print(llvm::outs()); }

void Value::printType(llvm::raw_ostream &Out) const {
  // Out << Interp->ValueTypeToString(*this);
}

void Value::printData(llvm::raw_ostream &Out) const {
  Out << Interp->ValueDataToString(*this);
}
// FIXME: We do not support the multiple inheritance case where one of the base
// classes has a pretty-printer and the other does not.
void Value::print(llvm::raw_ostream &Out) const {
  assert(OpaqueType != nullptr && "Can't print default Value");

  // Don't even try to print a void or an invalid type, it doesn't make sense.
  if (getType()->isVoidType() || !isValid())
    return;

  // We need to get all the results together then print it, since `printType` is
  // much faster than `printData`.
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  SS << "(";
  printType(SS);
  SS << ") ";
  printData(SS);
  SS << "\n";
  Out << Str;
}

class BuiltinValueBuffer : public ValueBuffer {
public:
  std::vector<char> raw;
  BuiltinValueBuffer(QualType _Ty) { Ty = _Ty; }
  template <typename T> T as() const {
    T v{};
    assert(raw.size() >= sizeof(T) && "Buffer too small for type!");
    memcpy(&v, raw.data(), sizeof(T));
    return v;
  }
  std::string toString() const override {
    if (Ty->isCharType()) {
      unsigned char c = as<unsigned char>();
      switch (c) {
      case '\n':
        return "'\\n'";
      case '\t':
        return "'\\t'";
      case '\r':
        return "'\\r'";
      case '\'':
        return "'\\''";
      case '\\':
        return "'\\'";
      default:
        if (std::isprint(c))
          return std::string("'") + static_cast<char>(c) + "'";
        else {
          return llvm::formatv("'\\x{0:02X}'", c).str();
        }
      }
    }
    if (auto *BT = Ty.getCanonicalType()->getAs<BuiltinType>()) {

      auto formatFloating = [](auto Val, char Suffix = '\0') -> std::string {
        std::string Out;
        llvm::raw_string_ostream SS(Out);

        if (std::isnan(Val) || std::isinf(Val)) {
          SS << llvm::format("%g", Val);
          return SS.str();
        }
        if (Val == static_cast<decltype(Val)>(static_cast<int64_t>(Val)))
          SS << llvm::format("%.1f", Val);
        else if (std::abs(Val) < 1e-4 || std::abs(Val) > 1e6 || Suffix == 'f')
          SS << llvm::format("%#.6g", Val);
        else if (Suffix == 'L')
          SS << llvm::format("%#.12Lg", Val);
        else
          SS << llvm::format("%#.8g", Val);

        if (Suffix != '\0')
          SS << Suffix;
        return SS.str();
      };

      std::string Str;
      llvm::raw_string_ostream SS(Str);
      switch (BT->getKind()) {
      default:
        return "{ error: unknown builtin type '" +
               std::to_string(BT->getKind()) + " '}";
      case clang::BuiltinType::Bool:
        SS << ((as<bool>()) ? "true" : "false");
        return Str;
      case clang::BuiltinType::Short:
        SS << as<short>();
        return Str;
      case clang::BuiltinType::UShort:
        SS << as<unsigned short>();
        return Str;
      case clang::BuiltinType::Int:
        SS << as<int>();
        return Str;
      case clang::BuiltinType::UInt:
        SS << as<unsigned int>();
        return Str;
      case clang::BuiltinType::Long:
        SS << as<long>();
        return Str;
      case clang::BuiltinType::ULong:
        SS << as<unsigned long>();
        return Str;
      case clang::BuiltinType::LongLong:
        SS << as<long long>();
        return Str;
      case clang::BuiltinType::ULongLong:
        SS << as<unsigned long long>();
        return Str;
      case clang::BuiltinType::Float:
        return formatFloating(as<float>(), /*suffix=*/'f');

      case clang::BuiltinType::Double:
        return formatFloating(as<double>());

      case clang::BuiltinType::LongDouble:
        return formatFloating(as<long double>(), /*suffix=*/'L');
      }
    }

    return "";
  }

  bool isValid() const override { return !raw.empty(); }
};

class ArrayValueBuffer : public ValueBuffer {
public:
  std::vector<std::unique_ptr<ValueBuffer>> Elements;
  ArrayValueBuffer(QualType EleTy) { Ty = EleTy; }
  std::string toString() const override {
    std::ostringstream OS;
    OS << "{";
    for (size_t i = 0; i < Elements.size(); ++i) {
      OS << Elements[i]->toString();
      if (i + 1 < Elements.size())
        OS << ",";
    }
    OS << "}";
    return OS.str();
  }

  bool isValid() const override { return !Elements.empty(); }
};

static std::string escapeString(const std::vector<char> &Raw) {
  std::string Out;
  for (char c : Raw) {
    switch (c) {
    case '\n':
      Out += "\\n";
      break;
    case '\t':
      Out += "\\t";
      break;
    case '\r':
      Out += "\\r";
      break;
    case '\"':
      Out += "\\\"";
      break;
    case '\\':
      Out += "\\\\";
      break;
    default:
      if (std::isprint(static_cast<unsigned char>(c)))
        Out.push_back(c);
      else {
        char buf[5];
        snprintf(buf, sizeof(buf), "\\x%02X", static_cast<unsigned char>(c));
        Out += buf;
      }
      break;
    }
  }
  return Out;
}

class PointerValueBuffer : public ValueBuffer {
public:
  uint64_t Address = 0;
  std::unique_ptr<ValueBuffer> Pointee; // optional, used only for char*

  PointerValueBuffer(QualType _Ty, uint64_t Addr = 0) : Address(Addr) {
    Ty = _Ty;
  }

  std::string toString() const override {
    auto PtrTy = dyn_cast<PointerType>(Ty.getTypePtr());
    if (!PtrTy)
      return "";

    auto PointeeTy = PtrTy->getPointeeType();

    // char* -> print string literal
    if (PointeeTy->isCharType() && Pointee) {
      if (auto *BE = static_cast<BuiltinValueBuffer *>(Pointee.get()))
        return "\"" + escapeString(BE->raw) + "\"";
    }

    if (Address == 0)
      return "nullptr";

    std::ostringstream OS;
    OS << "0x" << std::hex << Address;
    return OS.str();
  }

  bool isValid() const override { return Address != 0; }
};

class ReaderDispatcher {
private:
  ASTContext &Ctx;
  llvm::orc::MemoryAccess &MA;

public:
  ReaderDispatcher(ASTContext &Ctx, llvm::orc::MemoryAccess &MA)
      : Ctx(Ctx), MA(MA) {}

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  read(QualType QT, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  readBuiltin(QualType Ty, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  readPointer(QualType Ty, llvm::orc::ExecutorAddr Addr);

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  readArray(QualType Ty, llvm::orc::ExecutorAddr Addr);

  // TODO: record, function, etc.
};

class TypeReadVisitor
    : public TypeVisitor<TypeReadVisitor,
                         llvm::Expected<std::unique_ptr<ValueBuffer>>> {
  ReaderDispatcher &Dispatcher;
  llvm::orc::ExecutorAddr Addr;

public:
  TypeReadVisitor(ReaderDispatcher &D, llvm::orc::ExecutorAddr A)
      : Dispatcher(D), Addr(A) {}

  llvm::Expected<std::unique_ptr<ValueBuffer>> VisitType(const Type *T) {
    return llvm::make_error<llvm::StringError>(
        "Unsupported type in ReaderDispatcher", llvm::inconvertibleErrorCode());
  }

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  VisitBuiltinType(const BuiltinType *BT) {
    return Dispatcher.readBuiltin(QualType(BT, 0), Addr);
  }

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  VisitPointerType(const PointerType *PT) {
    return Dispatcher.readPointer(QualType(PT, 0), Addr);
  }

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  VisitConstantArrayType(const ConstantArrayType *AT) {
    return Dispatcher.readArray(QualType(AT, 0), Addr);
  }

  llvm::Expected<std::unique_ptr<ValueBuffer>>
  VisitRecordType(const RecordType *RT) {
    return llvm::make_error<llvm::StringError>(
        "RecordType reading not yet implemented",
        llvm::inconvertibleErrorCode());
  }
};

llvm::Expected<std::unique_ptr<ValueBuffer>>
ReaderDispatcher::read(QualType QT, llvm::orc::ExecutorAddr Addr) {
  TypeReadVisitor V(*this, Addr);
  return V.Visit(QT.getTypePtr());
}

llvm::Expected<std::unique_ptr<ValueBuffer>>
ReaderDispatcher::readBuiltin(QualType Ty, llvm::orc::ExecutorAddr Addr) {
  auto Size = Ctx.getTypeSizeInChars(Ty).getQuantity();
  auto ResOrErr = MA.readBuffers({llvm::orc::ExecutorAddrRange(Addr, Size)});
  if (!ResOrErr)
    return ResOrErr.takeError();

  auto Buf = std::make_unique<BuiltinValueBuffer>(Ty);
  const auto &Res = *ResOrErr;
  std::vector<char> ElemBuf(Size);
  std::memcpy(ElemBuf.data(), Res.back().data(), Size);
  Buf->raw = std::move(ElemBuf);
  return std::move(Buf);
}

llvm::Expected<std::unique_ptr<ValueBuffer>>
ReaderDispatcher::readArray(QualType Ty, llvm::orc::ExecutorAddr Addr) {
  const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(Ty);
  if (!CAT)
    return llvm::make_error<llvm::StringError>("Not a ConstantArrayType",
                                               llvm::inconvertibleErrorCode());

  QualType ElemTy = CAT->getElementType();
  size_t ElemSize = Ctx.getTypeSizeInChars(ElemTy).getQuantity();

  auto Buf = std::make_unique<ArrayValueBuffer>(Ty);

  for (size_t i = 0; i < CAT->getZExtSize(); ++i) {
    auto ElemAddr = Addr + i * ElemSize;
    auto ElemBufOrErr = read(ElemTy, ElemAddr);
    if (!ElemBufOrErr)
      return ElemBufOrErr.takeError();
    Buf->Elements.push_back(std::move(*ElemBufOrErr));
  }

  return std::move(Buf);
}

llvm::Expected<std::unique_ptr<ValueBuffer>>
ReaderDispatcher::ReaderDispatcher::readPointer(QualType Ty,
                                                llvm::orc::ExecutorAddr Addr) {
  auto PtrTy = dyn_cast<PointerType>(Ty.getTypePtr());
  if (!PtrTy)
    return llvm::make_error<llvm::StringError>("Not a PointerType",
                                               llvm::inconvertibleErrorCode());

  auto AddrOrErr = MA.readUInt64s({Addr});
  if (!AddrOrErr)
    return AddrOrErr.takeError();

  uint64_t PtrValAddr = AddrOrErr->back();
  if (PtrValAddr == 0)
    return std::make_unique<PointerValueBuffer>(Ty); // null pointer

  llvm::orc::ExecutorAddr PointeeAddr(PtrValAddr);
  auto PtrBuf = std::make_unique<PointerValueBuffer>(Ty, PtrValAddr);

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
    auto Buf = std::make_unique<BuiltinValueBuffer>(PointeeTy);
    Buf->raw.assign(S.begin(), S.end());
    PtrBuf->Pointee = std::move(Buf);
  }
  // else {
  //   auto BufOrErr = read(PointeeTy, PointeeAddr);
  //   if (!BufOrErr)
  //     return BufOrErr.takeError();
  //   PtrBuf->Pointee = std::move(*BufOrErr);
  // }
  return std::move(PtrBuf);
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
    Ty = It->second;
    IdToType.erase(It);
  }

  ReaderDispatcher Runner(Ctx, MemAcc);
  auto BufOrErr = Runner.read(Ty, Addr);

  ValBuf.reset();
  if (!BufOrErr) {
    SendResult(BufOrErr.takeError());
    return;
  }

  // Store the successfully read value buffer
  ValBuf.swap(*BufOrErr);

  SendResult(llvm::Error::success());
  return;
}
} // namespace clang
