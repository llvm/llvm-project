//===--- ExprUtils.h - Shared expression emission queries -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries about expressions that both classic CodeGen
// and CIR CodeGen need while emitting scalar, aggregate and lvalue
// expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H
#define LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"

namespace clang::CodeGenUtils {

/// Strip off the variably-modified array types wrapping \p VLA and return the
/// first element type that has a fixed size.
QualType getFixedSizeElementType(const ASTContext &Ctx,
                                 const VariableArrayType *VLA);

/// Check whether the value of \p E is possibly a reference to or into a
/// __block variable.
bool isBlockVarRef(const Expr *E);

/// Check whether \p E is cheap enough and side-effect-free enough to evaluate
/// unconditionally instead of conditionally.  This is used to convert control
/// flow into selects in some cases.
bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                            const ASTContext &Ctx);

/// Check whether \p E is a trivial array filler, that is, one that is
/// equivalent to zero-initialization.
bool isTrivialFiller(const Expr *E);

/// Determine whether the given cast kind is known to always convert values
/// with all zero bits in their value representation to values with all zero
/// bits in their value representation.
bool castPreservesZero(const CastExpr *CE);

/// isSimpleZero - If emitting this value will obviously just cause a store of
/// zero to memory, return true.  This can return false if uncertain, so it just
/// handles simple cases.
template <typename CodeGenFunctionTy>
bool isSimpleZero(const Expr *E, CodeGenFunctionTy &CGF) {
  E = E->IgnoreParens();
  while (auto *CE = dyn_cast<CastExpr>(E)) {
    if (!castPreservesZero(CE))
      break;
    E = CE->getSubExpr()->IgnoreParens();
  }

  // 0
  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(E))
    return IL->getValue() == 0;
  // +0.0
  if (const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(E))
    return FL->getValue().isPosZero();
  // int()
  if ((isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) &&
      CGF.getTypes().isZeroInitializable(E->getType()))
    return true;
  // (int*)0 - Null pointer expressions.
  if (const CastExpr *ICE = dyn_cast<CastExpr>(E))
    return ICE->getCastKind() == CK_NullToPointer &&
           CGF.getTypes().isPointerZeroInitializable(E->getType()) &&
           !E->HasSideEffects(CGF.getContext());
  // '\0'
  if (const CharacterLiteral *CL = dyn_cast<CharacterLiteral>(E))
    return CL->getValue() == 0;

  // Otherwise, hard case: conservatively return false.
  return false;
}

/// GetNumNonZeroBytesInInit - Get an approximate count of the number of
/// non-zero bytes that will be stored when outputting the initializer for the
/// specified initializer expression.
template <typename CodeGenFunctionTy>
CharUnits GetNumNonZeroBytesInInit(const Expr *E, CodeGenFunctionTy &CGF) {
  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();
  E = E->IgnoreParenNoopCasts(CGF.getContext());

  // 0 and 0.0 won't require any non-zero stores!
  if (isSimpleZero(E, CGF))
    return CharUnits::Zero();

  // If this is an initlist expr, sum up the size of sizes of the (present)
  // elements.  If this is something weird, assume the whole thing is non-zero.
  const InitListExpr *ILE = dyn_cast<InitListExpr>(E);
  while (ILE && ILE->isTransparent())
    ILE = dyn_cast<InitListExpr>(ILE->getInit(0));
  if (!ILE || !CGF.getTypes().isZeroInitializable(ILE->getType()))
    return CGF.getContext().getTypeSizeInChars(E->getType());

  // InitListExprs for structs have to be handled carefully.  If there are
  // reference members, we need to consider the size of the reference, not the
  // referencee.  InitListExprs for unions and arrays can't have references.
  if (const RecordType *RT = E->getType()->getAsCanonical<RecordType>()) {
    if (!RT->isUnionType()) {
      RecordDecl *SD = RT->getDecl()->getDefinitionOrSelf();
      CharUnits NumNonZeroBytes = CharUnits::Zero();

      unsigned ILEElement = 0;
      if (auto *CXXRD = dyn_cast<CXXRecordDecl>(SD))
        while (ILEElement != CXXRD->getNumBases())
          NumNonZeroBytes +=
              GetNumNonZeroBytesInInit(ILE->getInit(ILEElement++), CGF);
      for (const auto *Field : SD->fields()) {
        // We're done once we hit the flexible array member or run out of
        // InitListExpr elements.
        if (Field->getType()->isIncompleteArrayType() ||
            ILEElement == ILE->getNumInits())
          break;
        if (Field->isUnnamedBitField())
          continue;

        const Expr *E = ILE->getInit(ILEElement++);

        // Reference values are always non-null and have the width of a pointer.
        if (Field->getType()->isReferenceType())
          NumNonZeroBytes += CGF.getContext().toCharUnitsFromBits(
              CGF.getTarget().getPointerWidth(LangAS::Default));
        else
          NumNonZeroBytes += GetNumNonZeroBytesInInit(E, CGF);
      }

      return NumNonZeroBytes;
    }
  }

  // FIXME: This overestimates the number of non-zero bytes for bit-fields.
  CharUnits NumNonZeroBytes = CharUnits::Zero();
  for (unsigned i = 0, e = ILE->getNumInits(); i != e; ++i)
    NumNonZeroBytes += GetNumNonZeroBytesInInit(ILE->getInit(i), CGF);
  return NumNonZeroBytes;
}

/// Detect the unusual situation where an inline version of a builtin is
/// shadowed by a non-inline version.  In that case we should pick the external
/// one everywhere.  That's GCC behavior too.
bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD);

/// Given an object of the canonical type \p Type, can we safely copy a value
/// out of it based on its initializer?
///
/// FIXME: Overly conservative: rejects all arrays, and requires C++ records
/// to be trivial rather than just trivially copyable.
bool isConstantEmittableObjectType(QualType Type);

/// How a reference to a variable can be emitted as a constant.
enum ConstantEmissionKind {
  CEK_None,
  CEK_AsReferenceOnly,
  CEK_AsValueOrReference,
  CEK_AsValueOnly
};

/// Can we constant-emit a load of a reference to a variable of the given
/// type?  This is different from predicates like
/// Decl::mightBeUsableInConstantExpressions because we do want it to apply
/// in situations that don't necessarily satisfy the language's rules
/// for this (e.g. C++'s ODR-use rules).  For example, we want to be able
/// to do this with const float variables even if those variables
/// aren't marked 'constexpr'.
///
/// FIXME: For references, checks the pointee type, not the bound object's.
ConstantEmissionKind checkVarTypeForConstantEmission(QualType Type);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H
