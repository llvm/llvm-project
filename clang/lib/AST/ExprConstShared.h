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

// Return true if immediate and the comparison flags are matching
static bool MatchesPredicate(const uint32_t Imm, const FPCompareFlags &F) {
  switch (Imm & 0x1F) {
  case 0x00: /* _CMP_EQ_OQ */
  case 0x10: /* _CMP_EQ_OS */
    return F.IsEq && !F.IsUnordered;
  case 0x01: /* _CMP_LT_OS */
  case 0x11: /* _CMP_LT_OQ */
    return F.IsLt && !F.IsUnordered;
  case 0x02: /* _CMP_LE_OS */
  case 0x12: /* _CMP_LE_OQ */
    return !F.IsGt && !F.IsUnordered;
  case 0x03: /* _CMP_UNORD_Q */
  case 0x13: /* _CMP_UNORD_S */
    return F.IsUnordered;
  case 0x04: /* _CMP_NEQ_UQ */
  case 0x14: /* _CMP_NEQ_US */
    return !F.IsEq || F.IsUnordered;
  case 0x05: /* _CMP_NLT_US */
  case 0x15: /* _CMP_NLT_UQ */
    return !F.IsLt || F.IsUnordered;
  case 0x06: /* _CMP_NLE_US */
  case 0x16: /* _CMP_NLE_UQ */
    return F.IsGt || F.IsUnordered;
  case 0x07: /* _CMP_ORD_Q */
  case 0x17: /* _CMP_ORD_S */
    return !F.IsUnordered;
  case 0x08: /* _CMP_EQ_UQ */
  case 0x18: /* _CMP_EQ_US */
    return F.IsEq || F.IsUnordered;
  case 0x09: /* _CMP_NGE_US */
  case 0x19: /* _CMP_NGE_UQ */
    return F.IsLt || F.IsUnordered;
  case 0x0a: /* _CMP_NGT_US */
  case 0x1a: /* _CMP_NGT_UQ */
    return !F.IsGt || F.IsUnordered;
  case 0x0b: /* _CMP_FALSE_OQ */
  case 0x1b: /* _CMP_FALSE_OS */
    return false;
  case 0x0c: /* _CMP_NEQ_OQ */
  case 0x1c: /* _CMP_NEQ_OS */
    return !F.IsEq && !F.IsUnordered;
  case 0x0d: /* _CMP_GE_OS */
  case 0x1d: /* _CMP_GE_OQ */
    return !F.IsLt && !F.IsUnordered;
  case 0x0e: /* _CMP_GT_OS */
  case 0x1e: /* _CMP_GT_OQ */
    return F.IsGt && !F.IsUnordered;
  case 0x0f: /* _CMP_TRUE_UQ */
  case 0x1f: /* _CMP_TRUE_US */
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
