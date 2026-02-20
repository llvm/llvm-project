//===--- ExprConstShared.h - Shared consetxpr functionality ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared functionality between the new constant expression
// interpreter (AST/ByteCode/) and the current one (ExprConstant.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_EXPRCONSTSHARED_H
#define LLVM_CLANG_LIB_AST_EXPRCONSTSHARED_H

#include "clang/Basic/TypeTraits.h"
#include "llvm/ADT/APFloat.h"
#include <cstdint>

namespace llvm {
class APFloat;
class APSInt;
class APInt;
} // namespace llvm

namespace clang {
class QualType;
class LangOptions;
class ASTContext;
class CharUnits;
class Expr;

struct FPCompareFlags {
  bool IsUnordered;
  bool IsEq;
  bool IsGt;
  bool IsLt;
};

// SSE/AVX floating-point comparison immediates
namespace X86CmpImm {
constexpr uint32_t CMP_EQ_OQ = 0x00; // Equal (ordered, quiet)
constexpr uint32_t CMP_LT_OS = 0x01; // Less than (ordered, signaling)
constexpr uint32_t CMP_LE_OS = 0x02; // Less than or equal (ordered, signaling)
constexpr uint32_t CMP_UNORD_Q = 0x03; // Unordered (quiet)
constexpr uint32_t CMP_NEQ_UQ = 0x04;  // Not equal (unordered, quiet)
constexpr uint32_t CMP_NLT_US = 0x05;  // Not less than (unordered, signaling)
constexpr uint32_t CMP_NLE_US =
    0x06; // Not less than or equal (unordered, signaling)
constexpr uint32_t CMP_ORD_Q = 0x07; // Ordered (quiet)
constexpr uint32_t CMP_EQ_UQ = 0x08; // Equal (unordered, quiet)
constexpr uint32_t CMP_NGE_US =
    0x09; // Not greater than or equal (unordered, signaling)
constexpr uint32_t CMP_NGT_US = 0x0A; // Not greater than (unordered, signaling)
constexpr uint32_t CMP_FALSE_OQ = 0x0B; // False (ordered, quiet)
constexpr uint32_t CMP_NEQ_OQ = 0x0C;   // Not equal (ordered, quiet)
constexpr uint32_t CMP_GE_OS =
    0x0D; // Greater than or equal (ordered, signaling)
constexpr uint32_t CMP_GT_OS = 0x0E;   // Greater than (ordered, signaling)
constexpr uint32_t CMP_TRUE_UQ = 0x0F; // True (unordered, quiet)

// Signaling variants (0x10-0x1F)
constexpr uint32_t CMP_EQ_OS = 0x10;   // Equal (ordered, signaling)
constexpr uint32_t CMP_LT_OQ = 0x11;   // Less than (ordered, quiet)
constexpr uint32_t CMP_LE_OQ = 0x12;   // Less than or equal (ordered, quiet)
constexpr uint32_t CMP_UNORD_S = 0x13; // Unordered (signaling)
constexpr uint32_t CMP_NEQ_US = 0x14;  // Not equal (unordered, signaling)
constexpr uint32_t CMP_NLT_UQ = 0x15;  // Not less than (unordered, quiet)
constexpr uint32_t CMP_NLE_UQ =
    0x16; // Not less than or equal (unordered, quiet)
constexpr uint32_t CMP_ORD_S = 0x17; // Ordered (signaling)
constexpr uint32_t CMP_EQ_US = 0x18; // Equal (unordered, signaling)
constexpr uint32_t CMP_NGE_UQ =
    0x19; // Not greater than or equal (unordered, quiet)
constexpr uint32_t CMP_NGT_UQ = 0x1A;   // Not greater than (unordered, quiet)
constexpr uint32_t CMP_FALSE_OS = 0x1B; // False (ordered, signaling)
constexpr uint32_t CMP_NEQ_OS = 0x1C;   // Not equal (ordered, signaling)
constexpr uint32_t CMP_GE_OQ = 0x1D;   // Greater than or equal (ordered, quiet)
constexpr uint32_t CMP_GT_OQ = 0x1E;   // Greater than (ordered, quiet)
constexpr uint32_t CMP_TRUE_US = 0x1F; // True (unordered, signaling)
} // namespace X86CmpImm

// Return true if immediate and the comparison flags are matching
static bool MatchesPredicate(const uint32_t Imm,
                             const llvm::APFloatBase::cmpResult CompareResult) {
  using CmpResult = llvm::APFloatBase::cmpResult;

  bool IsUnordered = (CompareResult == llvm::APFloatBase::cmpUnordered);
  bool IsEq = (CompareResult == CmpResult::cmpEqual);
  bool IsGt = (CompareResult == CmpResult::cmpGreaterThan);
  bool IsLt = (CompareResult == CmpResult::cmpLessThan);

  switch (Imm & 0x1F) {
  case X86CmpImm::CMP_EQ_OQ:
  case X86CmpImm::CMP_EQ_OS:
    return IsEq && !IsUnordered;
  case X86CmpImm::CMP_LT_OS:
  case X86CmpImm::CMP_LT_OQ:
    return IsLt && !IsUnordered;
  case X86CmpImm::CMP_LE_OS:
  case X86CmpImm::CMP_LE_OQ:
    return !IsGt && !IsUnordered;
  case X86CmpImm::CMP_UNORD_Q:
  case X86CmpImm::CMP_UNORD_S:
    return IsUnordered;
  case X86CmpImm::CMP_NEQ_UQ:
  case X86CmpImm::CMP_NEQ_US:
    return !IsEq || IsUnordered;
  case X86CmpImm::CMP_NLT_US:
  case X86CmpImm::CMP_NLT_UQ:
    return !IsLt || IsUnordered;
  case X86CmpImm::CMP_NLE_US:
  case X86CmpImm::CMP_NLE_UQ:
    return IsGt || IsUnordered;
  case X86CmpImm::CMP_ORD_Q:
  case X86CmpImm::CMP_ORD_S:
    return !IsUnordered;
  case X86CmpImm::CMP_EQ_UQ:
  case X86CmpImm::CMP_EQ_US:
    return IsEq || IsUnordered;
  case X86CmpImm::CMP_NGE_US:
  case X86CmpImm::CMP_NGE_UQ:
    return IsLt || IsUnordered;
  case X86CmpImm::CMP_NGT_US:
  case X86CmpImm::CMP_NGT_UQ:
    return !IsGt || IsUnordered;
  case X86CmpImm::CMP_FALSE_OQ:
  case X86CmpImm::CMP_FALSE_OS:
    return false;
  case X86CmpImm::CMP_NEQ_OQ:
  case X86CmpImm::CMP_NEQ_OS:
    return !IsEq && !IsUnordered;
  case X86CmpImm::CMP_GE_OS:
  case X86CmpImm::CMP_GE_OQ:
    return !IsLt && !IsUnordered;
  case X86CmpImm::CMP_GT_OS:
  case X86CmpImm::CMP_GT_OQ:
    return IsGt && !IsUnordered;
  case X86CmpImm::CMP_TRUE_UQ:
  case X86CmpImm::CMP_TRUE_US:
    return true;
  }
  return false;
};
} // namespace clang

using namespace clang;
/// Values returned by __builtin_classify_type, chosen to match the values
/// produced by GCC's builtin.
enum class GCCTypeClass {
  None = -1,
  Void = 0,
  Integer = 1,
  // GCC reserves 2 for character types, but instead classifies them as
  // integers.
  Enum = 3,
  Bool = 4,
  Pointer = 5,
  // GCC reserves 6 for references, but appears to never use it (because
  // expressions never have reference type, presumably).
  PointerToDataMember = 7,
  RealFloat = 8,
  Complex = 9,
  // GCC reserves 10 for functions, but does not use it since GCC version 6 due
  // to decay to pointer. (Prior to version 6 it was only used in C++ mode).
  // GCC claims to reserve 11 for pointers to member functions, but *actually*
  // uses 12 for that purpose, same as for a class or struct. Maybe it
  // internally implements a pointer to member as a struct?  Who knows.
  PointerToMemberFunction = 12, // Not a bug, see above.
  ClassOrStruct = 12,
  Union = 13,
  // GCC reserves 14 for arrays, but does not use it since GCC version 6 due to
  // decay to pointer. (Prior to version 6 it was only used in C++ mode).
  // GCC reserves 15 for strings, but actually uses 5 (pointer) for string
  // literals.
  // Lang = 16,
  // OpaqueType = 17,
  BitInt = 18,
  Vector = 19
};

GCCTypeClass EvaluateBuiltinClassifyType(QualType T,
                                         const LangOptions &LangOpts);

void HandleComplexComplexMul(llvm::APFloat A, llvm::APFloat B, llvm::APFloat C,
                             llvm::APFloat D, llvm::APFloat &ResR,
                             llvm::APFloat &ResI);
void HandleComplexComplexDiv(llvm::APFloat A, llvm::APFloat B, llvm::APFloat C,
                             llvm::APFloat D, llvm::APFloat &ResR,
                             llvm::APFloat &ResI);

CharUnits GetAlignOfExpr(const ASTContext &Ctx, const Expr *E,
                         UnaryExprOrTypeTrait ExprKind);

uint8_t GFNIMultiplicativeInverse(uint8_t Byte);
uint8_t GFNIMul(uint8_t AByte, uint8_t BByte);
uint8_t GFNIAffine(uint8_t XByte, const llvm::APInt &AQword,
                   const llvm::APSInt &Imm, bool Inverse = false);
llvm::APSInt NormalizeRotateAmount(const llvm::APSInt &Value,
                                   const llvm::APSInt &Amount);

#endif
