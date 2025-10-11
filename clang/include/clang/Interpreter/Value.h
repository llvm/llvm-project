//===--- Value.h - Definition of interpreter value --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value is a lightweight struct that is used for carrying execution results in
// clang-repl. It's a special runtime that acts like a messager between compiled
// code and interpreted code. This makes it possible to exchange interesting
// information between the compiled & interpreted world.
//
// A typical usage is like the below:
//
// Value V;
// Interp.ParseAndExecute("int x = 42;");
// Interp.ParseAndExecute("x", &V);
// V.getType(); // <-- Yields a clang::QualType.
// V.getInt(); // <-- Yields 42.
//
// The current design is still highly experimental and nobody should rely on the
// API being stable because we're hopefully going to make significant changes to
// it in the relatively near future. For example, Value also intends to be used
// as an exchange token for JIT support enabling remote execution on the embed
// devices where the JIT infrastructure cannot fit. To support that we will need
// to split the memory storage in a different place and perhaps add a resource
// header is similar to intrinsics headers which have stricter performance
// constraints.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_VALUE_H
#define LLVM_CLANG_INTERPRETER_VALUE_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Config/llvm-config.h" // for LLVM_BUILD_LLVM_DYLIB, LLVM_BUILD_SHARED_LIBS
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>
#include <mutex>

// NOTE: Since the REPL itself could also include this runtime, extreme caution
// should be taken when MAKING CHANGES to this file, especially when INCLUDE NEW
// HEADERS, like <string>, <memory> and etc. (That pulls a large number of
// tokens and will impact the runtime performance of the REPL)

namespace llvm {
class raw_ostream;
namespace orc {
class ExecutionSession;
class LLJIT;
class MemoryAccess;
} // namespace orc
} // namespace llvm

namespace clang {

class ASTContext;
class Interpreter;
class QualType;

#if defined(_WIN32)
// REPL_EXTERNAL_VISIBILITY are symbols that we need to be able to locate
// at runtime. On Windows, this requires them to be exported from any of the
// modules loaded at runtime. Marking them as dllexport achieves this; both
// for DLLs (that normally export symbols as part of their interface) and for
// EXEs (that normally don't export anything).
// For a build with libclang-cpp.dll, this doesn't make any difference - the
// functions would have been exported anyway. But for cases when these are
// statically linked into an EXE, it makes sure that they're exported.
#define REPL_EXTERNAL_VISIBILITY __declspec(dllexport)
#elif __has_attribute(visibility)
#if defined(LLVM_BUILD_LLVM_DYLIB) || defined(LLVM_BUILD_SHARED_LIBS)
#define REPL_EXTERNAL_VISIBILITY __attribute__((visibility("default")))
#else
#define REPL_EXTERNAL_VISIBILITY
#endif
#else
#define REPL_EXTERNAL_VISIBILITY
#endif

#define REPL_BUILTIN_TYPES                                                     \
  X(bool, Bool)                                                                \
  X(char, Char_S)                                                              \
  X(signed char, SChar)                                                        \
  X(unsigned char, Char_U)                                                     \
  X(unsigned char, UChar)                                                      \
  X(short, Short)                                                              \
  X(unsigned short, UShort)                                                    \
  X(int, Int)                                                                  \
  X(unsigned int, UInt)                                                        \
  X(long, Long)                                                                \
  X(unsigned long, ULong)                                                      \
  X(long long, LongLong)                                                       \
  X(unsigned long long, ULongLong)                                             \
  X(float, Float)                                                              \
  X(double, Double)                                                            \
  X(long double, LongDouble)

class Value;

/// \struct ValueCleanup
/// \brief Encapsulates destructor invocation for REPL values.
///
/// `ValueCleanup` provides the logic to run object destructors in the JIT
/// process. It captures the runtime addresses of the destructor wrapper
/// functions and the object destructor itself.
///
/// Typical usage:
///  - Constructed when a JIT'd type requires cleanup.
///  - Attached to a `Value` via `setValueCleanup`.
///  - Invoked through `operator()(Value&)` to run the destructor on demand.
struct ValueCleanup {
  using DtorLookupFn =
      std::function<llvm::Expected<llvm::orc::ExecutorAddr>(QualType Ty)>;
  llvm::orc::ExecutionSession *ES;
  llvm::orc::ExecutorAddr DtorWrapperFn;
  llvm::orc::ExecutorAddr DtorFn;
  DtorLookupFn ObjDtor;
  ValueCleanup() = default;
  ValueCleanup(llvm::orc::ExecutionSession *ES,
               llvm::orc::ExecutorAddr WrapperFn,
               llvm::orc::ExecutorAddr DtorFn, DtorLookupFn Dtor)
      : ES(ES), DtorWrapperFn(WrapperFn), DtorFn(DtorFn),
        ObjDtor(std::move(Dtor)) {}
  ~ValueCleanup() = default;
  void operator()(Value &V);
};

/// \class Value
/// \brief Represents a dynamically typed value in the REPL.
///
/// `Value` provides a type-erased container for runtime values that can be
/// produced or consumed by the REPL. It supports multiple storage kinds:
///
///  - Builtin scalars (int, float, etc.)
///  - Arrays of `Value`
///  - Pointers (with optional pointee tracking)
///  - Strings
///  - An empty state (`K_None`)
///
/// `Value` also integrates with `ValueCleanup`, which holds runtime
/// destructor logic for objects that require cleanup.
class REPL_EXTERNAL_VISIBILITY Value final {
public:
  enum BuiltinKind {
#define X(type, name) K_##name,
    REPL_BUILTIN_TYPES
#undef X
        K_Unspecified
  };

private:
  /// Storage for builtin scalar values.
  struct Builtins {
  private:
    BuiltinKind BK = K_Unspecified;
    union {
#define X(type, name) type m_##name;
      REPL_BUILTIN_TYPES
#undef X
    };

  public:
    Builtins() = default;
    explicit Builtins(BuiltinKind BK) : BK(BK) {}
    ~Builtins() {}

    void setKind(BuiltinKind K) {
      assert(BK == K_Unspecified);
      BK = K;
    }
    BuiltinKind getKind() const { return BK; }
#define X(type, name)                                                          \
  void set##name(type Val) {                                                   \
    assert(BK == K_Unspecified || BK == K_##name);                             \
    m_##name = Val;                                                            \
    BK = K_##name;                                                             \
  }                                                                            \
  type get##name() const {                                                     \
    assert(BK != K_Unspecified);                                               \
    return m_##name;                                                           \
  }
    REPL_BUILTIN_TYPES
#undef X

    Builtins(const Builtins &) = delete;
    Builtins &operator=(const Builtins &) = delete;
  };

  /// Represents an array of `Value` elements.
  struct ArrValue {
    Value *Elements;
    uint64_t ArrSize;
    ArrValue(uint64_t Size) : Elements(new Value[Size]), ArrSize(Size) {}
    ~ArrValue() { delete[] Elements; }
    ArrValue(const ArrValue &) = delete;
    ArrValue &operator=(const ArrValue &) = delete;
  };

  /// Represents a pointer. Holds the address and optionally a pointee `Value`.
  struct PtrValue {
    uint64_t Addr = 0;
    Value *Pointee = nullptr; // optional for str
    PtrValue(uint64_t Addr) : Addr(Addr), Pointee(new Value()) {}
    ~PtrValue() {
      if (Pointee != nullptr)
        delete Pointee;
    }

    PtrValue(const PtrValue &) = delete;
    PtrValue &operator=(const PtrValue &) = delete;
  };

  /// Represents a string value (wrapper over std::string).
  struct StrValue {
    char *Buf;
    size_t Length;

    StrValue(const char *Str) {
      Length = strlen(Str);
      Buf = new char[Length + 1];
      memcpy(Buf, Str, Length);
      Buf[Length] = '\0';
    }

    ~StrValue() { delete[] Buf; }

    StrValue(const StrValue &) = delete;
    StrValue &operator=(const StrValue &) = delete;

    void set(const char *Str) {
      delete[] Buf;
      Length = strlen(Str);
      Buf = new char[Length + 1];
      memcpy(Buf, Str, Length);
      Buf[Length] = '\0';
    }

    const char *get() const { return Buf; }
  };

public:
  using DataType =
      llvm::AlignedCharArrayUnion<ArrValue, PtrValue, Builtins, StrValue>;
  enum ValKind { K_Builtin, K_Array, K_Pointer, K_Str, K_None };

private:
  QualType Ty;
  ValKind VKind = K_None;
  DataType Data;

  /// Optional cleanup action (e.g. call dtor in JIT runtime).
  std::optional<ValueCleanup> Cleanup = std::nullopt;

public:
  Value() : VKind(K_None) {}
  explicit Value(QualType Ty) : Ty(Ty), VKind(K_None) {}
  Value(const Value &RHS);
  Value(Value &&RHS)
      : Ty(RHS.Ty), VKind(RHS.VKind), Data(RHS.Data),
        Cleanup(std::move(RHS.Cleanup)) {
    RHS.VKind = K_None;
  }

  Value &operator=(const Value &RHS);
  Value &operator=(Value &&RHS);

  explicit Value(QualType QT, std::vector<uint8_t> Raw);

  struct UninitArr {};

  explicit Value(UninitArr, QualType QT, uint64_t ArrSize)
      : Ty(QT), VKind(K_None) {
    MakeArray(ArrSize);
  }

  explicit Value(QualType QT, uint64_t Addr) : Ty(QT), VKind(K_None) {
    MakePointer(Addr);
  }

  explicit Value(QualType QT, const char *buf) : Ty(QT), VKind(K_None) {
    MakeStr(buf);
  }

  ~Value() {
    if (VKind != K_None)
      destroy();
  }

  // ---- Raw buffer conversion ----
  template <typename T> static T as(std::vector<uint8_t> &raw) {
    T v{};
    // assert(raw.size() >= sizeof(T) && "Buffer too small for type!");
    memcpy(&v, raw.data(), sizeof(T));
    return v;
  }

  // ---- Kind checks ----
  bool isUnknown() const { return VKind == K_None; }
  bool isBuiltin() const { return VKind == K_Builtin; }
  bool isArray() const { return VKind == K_Array; }
  bool isPointer() const { return VKind == K_Pointer; }
  bool isStr() const { return VKind == K_Str; }
  ValKind getKind() const { return VKind; }
  QualType getType() const { return Ty; }
  bool isAbsent() const { return VKind == K_None; }
  bool hasValue() const { return VKind != K_None; }
  BuiltinKind getBuiltinKind() const {
    if (isBuiltin())
      return asBuiltin().getKind();
    return BuiltinKind::K_Unspecified;
  }

protected:
  // ---- accessors ----
  Builtins &asBuiltin() {
    assert(isBuiltin() && "Not a builtin value");
    return *((Builtins *)(char *)&Data);
  }

  const Builtins &asBuiltin() const {
    return const_cast<Value *>(this)->asBuiltin();
  }

  ArrValue &asArray() {
    assert(isArray() && "Not an array value");
    return *((ArrValue *)(char *)&Data);
  }

  const ArrValue &asArray() const {
    return const_cast<Value *>(this)->asArray();
  }

  PtrValue &asPointer() {
    assert(isPointer() && "Not a pointer value");
    return *((PtrValue *)(char *)&Data);
  }

  const PtrValue &asPointer() const {
    return const_cast<Value *>(this)->asPointer();
  }

  StrValue &asStr() {
    assert(isStr() && "Not a Str value");
    return *((StrValue *)(char *)&Data);
  }

  const StrValue &asStr() const { return const_cast<Value *>(this)->asStr(); }

public:
  // ---- BuiltinKind Query helpers ----
  bool hasBuiltinThis(BuiltinKind K) const {
    if (isBuiltin())
      return asBuiltin().getKind() == K;
    return false;
  }

  // ---- String accessors ----
  void setStrVal(const char *buf) {
    assert(isStr() && "Not a Str");
    asStr().set(buf);
  }

  const char *getStrVal() const {
    assert(isStr() && "Not a Str");
    return asStr().get();
  }

  // ---- Array accessors ----
  uint64_t getArraySize() const { return asArray().ArrSize; }

  uint64_t getArrayInitializedElts() const { return asArray().ArrSize; }

  Value &getArrayInitializedElt(unsigned I) {
    assert(isArray() && "Invalid accessor");
    assert(I < getArrayInitializedElts() && "Index out of range");
    return ((ArrValue *)(char *)&Data)->Elements[I];
  }

  const Value &getArrayInitializedElt(unsigned I) const {
    return const_cast<Value *>(this)->getArrayInitializedElt(I);
  }

  // ---- Pointer accessors ----
  bool HasPointee() const {
    assert(isPointer() && "Invalid accessor");
    return !(asPointer().Pointee->isAbsent());
  }

  Value &getPointerPointee() {
    assert(isPointer() && "Invalid accessor");
    return *asPointer().Pointee;
  }

  const Value &getPointerPointee() const {
    return const_cast<Value *>(this)->getPointerPointee();
  }

  uint64_t getAddr() const { return asPointer().Addr; }

  // ---- Builtin setters/getters ----
#define X(type, name)                                                          \
  void set##name(type Val) { asBuiltin().set##name(Val); }                     \
  type get##name() const { return asBuiltin().get##name(); }
  REPL_BUILTIN_TYPES
#undef X

  // ---- Printing ----
  void printType(llvm::raw_ostream &Out, ASTContext &Ctx) const;
  void printData(llvm::raw_ostream &Out, ASTContext &Ctx) const;
  void print(llvm::raw_ostream &Out, ASTContext &Ctx) const;
  void dump(ASTContext &Ctx) const;

  // ---- Cleanup & destruction ----
  void setValueCleanup(ValueCleanup VC) {
    assert(!Cleanup.has_value());
    Cleanup.emplace(std::move(VC));
  }

  bool hasAttachedCleanup() const { return Cleanup.has_value(); }

  void clear() {
    if (Cleanup.has_value())
      (*Cleanup)(*this);
    destroy();
  }

private:
  // ---- Constructors for each kind ----
  void MakeBuiltIns() {
    assert(isAbsent() && "Bad state change");
    new ((void *)(char *)&Data) Builtins(BuiltinKind::K_Unspecified);
    VKind = K_Builtin;
  }

  void MakeArray(uint64_t Size) {
    assert(isAbsent() && "Bad state change");
    new ((void *)(char *)&Data) ArrValue(Size);
    VKind = K_Array;
  }

  void MakePointer(uint64_t Addr = 0) {
    assert(isAbsent() && "Bad state change");
    new ((void *)(char *)&Data) PtrValue(Addr);
    VKind = K_Pointer;
  }

  void MakeStr(const char *Str) {
    assert(isAbsent() && "Bad state change");
    new ((void *)(char *)&Data) StrValue(Str);
    VKind = K_Str;
  }

  void setBuiltins(Builtins &LHS, const Builtins &RHS) {
    switch (RHS.getKind()) {
    default:
      assert(false && "Type not supported");

#define X(type, name)                                                          \
  case BuiltinKind::K_##name: {                                                \
    LHS.setKind(BuiltinKind::K_##name);                                        \
    LHS.set##name(RHS.get##name());                                            \
  } break;
      REPL_BUILTIN_TYPES
#undef X
    }
  }

  void destroy() {
    switch (VKind) {
    case K_Builtin:
      ((Builtins *)(char *)&Data)->~Builtins();
      break;
    case K_Array:
      ((ArrValue *)(char *)&Data)->~ArrValue();
      break;
    case K_Pointer:
      ((PtrValue *)(char *)&Data)->~PtrValue();
      break;
    case K_Str:
      ((StrValue *)(char *)&Data)->~StrValue();
      break;
    default:
      break;
    }
    VKind = K_None;
  }
};

class ValueToString {
private:
  ASTContext &Ctx;

public:
  ValueToString(ASTContext &Ctx) : Ctx(Ctx) {}
  std::string toString(const Value *);
  std::string toString(QualType);

private:
  std::string BuiltinToString(const Value &B);
  std::string PointerToString(const Value &P);
  std::string ArrayToString(const Value &A);
};

/// \class ValueResultManager
/// \brief Manages values returned from JIT code.
///
/// Each result is registered with a unique ID and its `QualType`.
/// The JIT code later calls back into the runtime with that ID, and
/// `deliverResult` uses it to look up the type, read the value from memory,
/// and attach any destructor cleanup before making it available to the host.
class ValueResultManager {
public:
  using ValueId = uint64_t;
  using SendResultFn = llvm::unique_function<void(llvm::Error)>;

  explicit ValueResultManager(ASTContext &Ctx, llvm::orc::MemoryAccess &MemAcc);

  static std::unique_ptr<ValueResultManager>
  Create(llvm::orc::LLJIT &EE, ASTContext &Ctx, bool IsOutOfProcess = false);

  ValueId registerPendingResult(QualType QT,
                                std::optional<ValueCleanup> VC = std::nullopt) {
    std::lock_guard<std::mutex> Lock(Mutex);
    ValueId NewID = NextID.fetch_add(1, std::memory_order_relaxed);
    IdToType.insert({NewID, QT});
    if (VC)
      IdToValCleanup.insert({NewID, std::move(*VC)});
    return NewID;
  }

  void resetAndDump();

  void deliverResult(SendResultFn SendResult, ValueId ID,
                     llvm::orc::ExecutorAddr VAddr);
  Value release() { return std::move(LastVal); }

private:
  std::atomic<ValueId> NextID{1};
  void Initialize(llvm::orc::LLJIT &EE);

  mutable std::mutex Mutex;
  ASTContext &Ctx;
  llvm::orc::MemoryAccess &MemAcc;
  Value LastVal;
  llvm::DenseMap<ValueId, QualType> IdToType;
  llvm::DenseMap<ValueId, ValueCleanup> IdToValCleanup;
};

} // namespace clang
#endif
