//===--- PrimType.h - Types for the constexpr VM --------------------*- C++ -*-===//
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

#include <climits>
#include <cstddef>
#include <cstdint>
#include "Boolean.h"
#include "Integral.h"

namespace clang {
namespace interp {

class Pointer;

/// Enumeration of the primitive types of the VM.
enum PrimType : unsigned {
  PT_Sint8,
  PT_Uint8,
  PT_Sint16,
  PT_Uint16,
  PT_Sint32,
  PT_Uint32,
  PT_Sint64,
  PT_Uint64,
  PT_Bool,
  PT_Ptr,
};

/// Mapping from primitive types to their representation.
template <PrimType T> struct PrimConv;
template <> struct PrimConv<PT_Sint8> { using T = Integral<8, true>; };
template <> struct PrimConv<PT_Uint8> { using T = Integral<8, false>; };
template <> struct PrimConv<PT_Sint16> { using T = Integral<16, true>; };
template <> struct PrimConv<PT_Uint16> { using T = Integral<16, false>; };
template <> struct PrimConv<PT_Sint32> { using T = Integral<32, true>; };
template <> struct PrimConv<PT_Uint32> { using T = Integral<32, false>; };
template <> struct PrimConv<PT_Sint64> { using T = Integral<64, true>; };
template <> struct PrimConv<PT_Uint64> { using T = Integral<64, false>; };
template <> struct PrimConv<PT_Bool> { using T = Boolean; };
template <> struct PrimConv<PT_Ptr> { using T = Pointer; };

/// Returns the size of a primitive type in bytes.
size_t primSize(PrimType Type);

/// Aligns a size to the pointer alignment.
constexpr size_t align(size_t Size) {
  return ((Size + alignof(void *) - 1) / alignof(void *)) * alignof(void *);
}

inline bool isPrimitiveIntegral(PrimType Type) {
  switch (Type) {
  case PT_Bool:
  case PT_Sint8:
  case PT_Uint8:
  case PT_Sint16:
  case PT_Uint16:
  case PT_Sint32:
  case PT_Uint32:
  case PT_Sint64:
  case PT_Uint64:
    return true;
  default:
    return false;
  }
}

} // namespace interp
} // namespace clang

/// Helper macro to simplify type switches.
/// The macro implicitly exposes a type T in the scope of the inner block.
#define TYPE_SWITCH_CASE(Name, B) \
  case Name: { using T = PrimConv<Name>::T; B; break; }
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
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
      TYPE_SWITCH_CASE(PT_Ptr, B)                                              \
    }                                                                          \
  } while (0)
#define COMPOSITE_TYPE_SWITCH(Expr, B, D)                                      \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Ptr, B)                                              \
      default: { D; break; }                                                   \
    }                                                                          \
  } while (0)
#endif
