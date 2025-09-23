//===--- PrimType.h - Types for the constexpr VM ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the VM types and helpers operating on types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_TYPE_H
#define LLVM_CLANG_AST_INTERP_TYPE_H

#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <cstddef>
#include <cstdint>

namespace clang {
namespace interp {

class Pointer;
class Boolean;
class Floating;
class FunctionPointer;
class MemberPointer;
class FixedPoint;
template <bool Signed> class IntegralAP;
template <unsigned Bits, bool Signed> class Integral;

/// Enumeration of the primitive types of the VM.
enum PrimType : uint8_t {
  PT_Sint8 = 0,
  PT_Uint8 = 1,
  PT_Sint16 = 2,
  PT_Uint16 = 3,
  PT_Sint32 = 4,
  PT_Uint32 = 5,
  PT_Sint64 = 6,
  PT_Uint64 = 7,
  PT_IntAP = 8,
  PT_IntAPS = 9,
  PT_Bool = 10,
  PT_FixedPoint = 11,
  PT_Float = 12,
  PT_Ptr = 13,
  PT_MemberPtr = 14,
};

// Like std::optional<PrimType>, but only sizeof(PrimType).
class OptPrimType final {
  static constexpr uint8_t None = 0xFF;
  uint8_t V = None;

public:
  OptPrimType() = default;
  OptPrimType(std::nullopt_t) {}
  OptPrimType(PrimType T) : V(static_cast<unsigned>(T)) {}

  explicit constexpr operator bool() const { return V != None; }
  PrimType operator*() const {
    assert(operator bool());
    return static_cast<PrimType>(V);
  }

  PrimType value_or(PrimType PT) const {
    if (operator bool())
      return static_cast<PrimType>(V);
    return PT;
  }

  bool operator==(PrimType PT) const {
    if (!operator bool())
      return false;
    return V == static_cast<unsigned>(PT);
  }
  bool operator==(OptPrimType OPT) const { return V == OPT.V; }
  bool operator!=(PrimType PT) const { return !(*this == PT); }
  bool operator!=(OptPrimType OPT) const { return V != OPT.V; }
};
static_assert(sizeof(OptPrimType) == sizeof(PrimType));

inline constexpr bool isPtrType(PrimType T) {
  return T == PT_Ptr || T == PT_MemberPtr;
}

inline constexpr bool isSignedType(PrimType T) {
  switch (T) {
  case PT_Sint8:
  case PT_Sint16:
  case PT_Sint32:
  case PT_Sint64:
    return true;
  default:
    return false;
  }
  return false;
}

enum class CastKind : uint8_t {
  Reinterpret,
  Volatile,
  Dynamic,
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     interp::CastKind CK) {
  switch (CK) {
  case interp::CastKind::Reinterpret:
    OS << "reinterpret_cast";
    break;
  case interp::CastKind::Volatile:
    OS << "volatile";
    break;
  case interp::CastKind::Dynamic:
    OS << "dynamic";
    break;
  }
  return OS;
}

constexpr bool isIntegralType(PrimType T) { return T <= PT_FixedPoint; }
template <typename T> constexpr bool needsAlloc() {
  return std::is_same_v<T, IntegralAP<false>> ||
         std::is_same_v<T, IntegralAP<true>> || std::is_same_v<T, Floating>;
}
constexpr bool needsAlloc(PrimType T) {
  return T == PT_IntAP || T == PT_IntAPS || T == PT_Float;
}

/// Mapping from primitive types to their representation.
template <PrimType T> struct PrimConv;
template <> struct PrimConv<PT_Sint8> {
  using T = Integral<8, true>;
};
template <> struct PrimConv<PT_Uint8> {
  using T = Integral<8, false>;
};
template <> struct PrimConv<PT_Sint16> {
  using T = Integral<16, true>;
};
template <> struct PrimConv<PT_Uint16> {
  using T = Integral<16, false>;
};
template <> struct PrimConv<PT_Sint32> {
  using T = Integral<32, true>;
};
template <> struct PrimConv<PT_Uint32> {
  using T = Integral<32, false>;
};
template <> struct PrimConv<PT_Sint64> {
  using T = Integral<64, true>;
};
template <> struct PrimConv<PT_Uint64> {
  using T = Integral<64, false>;
};
template <> struct PrimConv<PT_IntAP> {
  using T = IntegralAP<false>;
};
template <> struct PrimConv<PT_IntAPS> {
  using T = IntegralAP<true>;
};
template <> struct PrimConv<PT_Float> {
  using T = Floating;
};
template <> struct PrimConv<PT_Bool> {
  using T = Boolean;
};
template <> struct PrimConv<PT_Ptr> {
  using T = Pointer;
};
template <> struct PrimConv<PT_MemberPtr> {
  using T = MemberPointer;
};
template <> struct PrimConv<PT_FixedPoint> {
  using T = FixedPoint;
};

/// Returns the size of a primitive type in bytes.
size_t primSize(PrimType Type);

/// Aligns a size to the pointer alignment.
constexpr size_t align(size_t Size) {
  return ((Size + alignof(void *) - 1) / alignof(void *)) * alignof(void *);
}

constexpr bool aligned(uintptr_t Value) { return Value == align(Value); }
static_assert(aligned(sizeof(void *)));

static inline bool aligned(const void *P) {
  return aligned(reinterpret_cast<uintptr_t>(P));
}

} // namespace interp
} // namespace clang

/// Helper macro to simplify type switches.
/// The macro implicitly exposes a type T in the scope of the inner block.
#define TYPE_SWITCH_CASE(Name, B)                                              \
  case Name: {                                                                 \
    using T = PrimConv<Name>::T;                                               \
    B;                                                                         \
    break;                                                                     \
  }
#define TYPE_SWITCH(Expr, B)                                                   \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Sint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Uint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Sint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint64, B)                                           \
      TYPE_SWITCH_CASE(PT_IntAP, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAPS, B)                                           \
      TYPE_SWITCH_CASE(PT_Float, B)                                            \
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
      TYPE_SWITCH_CASE(PT_Ptr, B)                                              \
      TYPE_SWITCH_CASE(PT_MemberPtr, B)                                        \
      TYPE_SWITCH_CASE(PT_FixedPoint, B)                                       \
    }                                                                          \
  } while (0)

#define INT_TYPE_SWITCH(Expr, B)                                               \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Sint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Uint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Sint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint64, B)                                           \
      TYPE_SWITCH_CASE(PT_IntAP, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAPS, B)                                           \
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
    default:                                                                   \
      llvm_unreachable("Not an integer value");                                \
    }                                                                          \
  } while (0)

#define INT_TYPE_SWITCH_NO_BOOL(Expr, B)                                       \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Sint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Uint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Sint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint64, B)                                           \
      TYPE_SWITCH_CASE(PT_IntAP, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAPS, B)                                           \
    default:                                                                   \
      llvm_unreachable("Not an integer value");                                \
    }                                                                          \
  } while (0)

#define TYPE_SWITCH_ALLOC(Expr, B)                                             \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Float, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAP, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAPS, B)                                           \
    default:;                                                                  \
    }                                                                          \
  } while (0)

#endif
