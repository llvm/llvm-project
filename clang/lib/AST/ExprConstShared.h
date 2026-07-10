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
#include <optional>

namespace llvm {
class APFloat;
class APSInt;
class APInt;
}
namespace clang {
class QualType;
class LangOptions;
class ASTContext;
class CharUnits;
class Expr;
class CallExpr;
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

/// Convert a builtin ID to the canonical x86 builtin ID the constant evaluators
/// dispatch on in their x86 target-specific cases.
///
/// Target-independent builtins are returned unchanged. An x86 target builtin
/// (including an auxiliary-target x86 builtin, whose ID is shifted past the
/// primary target's builtins) is translated to its canonical X86::BI* value.
/// Any other target's builtin returns 0: the constant evaluators only fold x86
/// target builtins, and target builtin IDs of different targets overlap (each
/// numbers from Builtin::FirstTSBuiltin), so an unrelated target's ID must not
/// be mistaken for an x86 one.
///
/// The ID-based overload performs no work beyond a single comparison for
/// target-independent builtins, so it is suitable for hot paths (e.g. the
/// bytecode interpreter's builtin dispatch) where re-deriving the ID from the
/// call expression would be wasteful.
unsigned ConvertBuiltinIDToX86BuiltinID(const ASTContext &Ctx,
                                        unsigned BuiltinID);
unsigned ConvertBuiltinIDToX86BuiltinID(const ASTContext &Ctx,
                                        const CallExpr *E);

uint8_t GFNIMultiplicativeInverse(uint8_t Byte);
uint8_t GFNIMul(uint8_t AByte, uint8_t BByte);
uint8_t GFNIAffine(uint8_t XByte, const llvm::APInt &AQword,
                   const llvm::APSInt &Imm, bool Inverse = false);
llvm::APSInt NormalizeRotateAmount(const llvm::APSInt &Value,
                                   const llvm::APSInt &Amount);

std::optional<llvm::APFloat>
EvalScalarMinMaxFp(const llvm::APFloat &A, const llvm::APFloat &B,
                   std::optional<llvm::APSInt> RoundingMode, bool IsMin);

#endif
