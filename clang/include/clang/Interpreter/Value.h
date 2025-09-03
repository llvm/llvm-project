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
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>

// NOTE: Since the REPL itself could also include this runtime, extreme caution
// should be taken when MAKING CHANGES to this file, especially when INCLUDE NEW
// HEADERS, like <string>, <memory> and etc. (That pulls a large number of
// tokens and will impact the runtime performance of the REPL)

namespace llvm {
class raw_ostream;

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

class REPL_EXTERNAL_VISIBILITY Value {
  union Storage {
#define X(type, name) type m_##name;
    REPL_BUILTIN_TYPES
#undef X
    void *m_Ptr;
    unsigned char m_RawBits[sizeof(long double) * 8]; // widest type
  };

public:
  enum Kind {
#define X(type, name) K_##name,
    REPL_BUILTIN_TYPES
#undef X

        K_Void,
    K_PtrOrObj,
    K_Unspecified
  };

  Value() = default;
  Value(const Interpreter *In, void *Ty);
  Value(const Value &RHS);
  Value(Value &&RHS) noexcept;
  Value &operator=(const Value &RHS);
  Value &operator=(Value &&RHS) noexcept;
  ~Value();

  void printType(llvm::raw_ostream &Out) const;
  void printData(llvm::raw_ostream &Out) const;
  void print(llvm::raw_ostream &Out) const;
  void dump() const;
  void clear();

  const ASTContext &getASTContext() const;
  const Interpreter &getInterpreter() const;
  QualType getType() const;

  bool isValid() const { return ValueKind != K_Unspecified; }
  bool isVoid() const { return ValueKind == K_Void; }
  bool hasValue() const { return isValid() && !isVoid(); }
  bool isManuallyAlloc() const { return IsManuallyAlloc; }
  Kind getKind() const { return ValueKind; }
  void setKind(Kind K) { ValueKind = K; }
  void setOpaqueType(void *Ty) { OpaqueType = Ty; }

  void *getPtr() const;
  void setPtr(void *Ptr) { Data.m_Ptr = Ptr; }
  void setRawBits(void *Ptr, unsigned NBits = sizeof(Storage));

#define X(type, name)                                                          \
  void set##name(type Val) { Data.m_##name = Val; }                            \
  type get##name() const { return Data.m_##name; }
  REPL_BUILTIN_TYPES
#undef X

  /// \brief Get the value with cast.
  //
  /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
  /// casting the value of builtins (except void), enums and pointers.
  /// Values referencing an object are treated as pointers to the object.
  template <typename T> T convertTo() const {
    return convertFwd<T>::cast(*this);
  }

protected:
  bool isPointerOrObjectType() const { return ValueKind == K_PtrOrObj; }

  /// \brief Get to the value with type checking casting the underlying
  /// stored value to T.
  template <typename T> T as() const {
    switch (ValueKind) {
    default:
      return T();
#define X(type, name)                                                          \
  case Value::K_##name:                                                        \
    return (T)Data.m_##name;
      REPL_BUILTIN_TYPES
#undef X
    }
  }

  // Allow convertTo to be partially specialized.
  template <typename T> struct convertFwd {
    static T cast(const Value &V) {
      if (V.isPointerOrObjectType())
        return (T)(uintptr_t)V.as<void *>();
      if (!V.isValid() || V.isVoid()) {
        return T();
      }
      return V.as<T>();
    }
  };

  template <typename T> struct convertFwd<T *> {
    static T *cast(const Value &V) {
      if (V.isPointerOrObjectType())
        return (T *)(uintptr_t)V.as<void *>();
      return nullptr;
    }
  };

  const Interpreter *Interp = nullptr;
  void *OpaqueType = nullptr;
  Storage Data;
  Kind ValueKind = K_Unspecified;
  bool IsManuallyAlloc = false;
};

template <> inline void *Value::as() const {
  if (isPointerOrObjectType())
    return Data.m_Ptr;
  return (void *)as<uintptr_t>();
}

class ValueBuffer {
public:
  enum Kind { K_Builtin, K_Array, K_Pointer, K_Unknown };
  QualType Ty;
  Kind K;
  ValueBuffer(Kind K = K_Unknown) : K(K) {}
  virtual ~ValueBuffer() = default;
  bool isUnknown() const { return K == K_Unknown; }
  bool isBuiltin() const { return K == K_Builtin; }
  bool isArray() const { return K == K_Array; }
  bool isPointer() const { return K == K_Pointer; }
  static bool classof(const ValueBuffer *) { return true; }
};

class BuiltinValueBuffer;
class ArrayValueBuffer;
class PointerValueBuffer;

class BuiltinValueBuffer : public ValueBuffer {
public:
  std::vector<uint8_t> raw;
  BuiltinValueBuffer(QualType _Ty) : ValueBuffer(K_Builtin) { Ty = _Ty; }
  template <typename T> T as() const {
    T v{};
    // assert(raw.size() >= sizeof(T) && "Buffer too small for type!");
    memcpy(&v, raw.data(), sizeof(T));
    return v;
  }
  static bool classof(const ValueBuffer *B) { return B->isBuiltin(); }
};

class ArrayValueBuffer : public ValueBuffer {
public:
  std::vector<std::unique_ptr<ValueBuffer>> Elements;
  ArrayValueBuffer(QualType EleTy) : ValueBuffer(K_Array) { Ty = EleTy; }

  static bool classof(const ValueBuffer *B) { return B->isArray(); }
};

class PointerValueBuffer : public ValueBuffer {
public:
  uint64_t Address = 0;
  std::unique_ptr<ValueBuffer> Pointee; // optional, used only for char*

  PointerValueBuffer(QualType _Ty, uint64_t Addr = 0)
      : ValueBuffer(K_Pointer), Address(Addr) {
    Ty = _Ty;
  }

  static bool classof(const ValueBuffer *B) { return B->isPointer(); }
};

class ValueToString {
private:
  ASTContext &Ctx;

public:
  ValueToString(ASTContext &Ctx) : Ctx(Ctx) {}
  std::string toString(const ValueBuffer *);

private:
  std::string BuiltinToString(const BuiltinValueBuffer &B);
  std::string PointerToString(const PointerValueBuffer &P);
  std::string ArrayToString(const ArrayValueBuffer &A);
};

class ValueResultManager {
public:
  using ValueId = uint64_t;
  using SendResultFn = llvm::unique_function<void(llvm::Error)>;

  explicit ValueResultManager(ASTContext &Ctx, llvm::orc::MemoryAccess &MemAcc);

  static std::unique_ptr<ValueResultManager>
  Create(llvm::orc::LLJIT &EE, ASTContext &Ctx, bool IsOutOfProcess = false);

  ValueId registerPendingResult(QualType QT) {
    ValueId NewID = NextID.fetch_add(1, std::memory_order_relaxed);
    IdToType.insert({NewID, QT});
    return NewID;
  }

  void resetAndDump();

  void deliverResult(SendResultFn SendResult, ValueId ID,
                     llvm::orc::ExecutorAddr VAddr);

private:
  std::atomic<ValueId> NextID{1};
  void Initialize(llvm::orc::LLJIT &EE);

  std::string ValueTypeToString(QualType QT) const;

  mutable std::mutex Mutex;
  ASTContext &Ctx;
  llvm::orc::MemoryAccess &MemAcc;
  std::unique_ptr<ValueBuffer> ValBuf = nullptr;
  llvm::DenseMap<ValueId, clang::QualType> IdToType;
};

} // namespace clang
#endif
