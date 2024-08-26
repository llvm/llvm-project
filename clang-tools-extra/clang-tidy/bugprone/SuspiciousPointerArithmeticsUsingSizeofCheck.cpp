//===--- SuspiciousPointerArithmeticsUsingSizeofCheck.cpp - clang-tidy ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousPointerArithmeticsUsingSizeofCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

//static const char *bin_op_bind = "ptr-sizeof-expression";	
static constexpr llvm::StringLiteral BinOp{"bin-op"};
static constexpr llvm::StringLiteral PointedType{"pointed-type"};
static const auto IgnoredType = qualType(anyOf(asString("char"),asString("unsigned char"),asString("signed char"),asString("int8_t"),asString("uint8_t"),asString("std::byte"),asString("const char"),asString("const unsigned char"),asString("const signed char"),asString("const int8_t"),asString("const uint8_t"),asString("const std::byte")));
static const auto InterestingPointer = pointerType(unless(pointee(IgnoredType)));

SuspiciousPointerArithmeticsUsingSizeofCheck::SuspiciousPointerArithmeticsUsingSizeofCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(
	     expr(anyOf(
/*                    binaryOperator(hasAnyOperatorName("+","-"),
                      hasEitherOperand(hasType(pointerType())),
		      hasEitherOperand(sizeOfExpr(expr())),
		      unless(allOf(hasLHS(hasType(pointerType())),
				   hasRHS(hasType(pointerType()))))
		      ).bind(bin_op_bind),
		    binaryOperator(hasAnyOperatorName("+=","-="),
	              hasLHS(hasType(pointerType())),
		      hasRHS(sizeOfExpr(expr()))
		      ).bind(bin_op_bind)

		    binaryOperator(hasAnyOperatorName("+=","-=","+","-" ),
	              hasLHS(hasType(InterestingPointer)),
		      hasRHS(sizeOfExpr(expr()))).bind(BinOp),
		    binaryOperator(hasAnyOperatorName("+","-" ),
	              hasRHS(hasType(InterestingPointer)),
		      hasLHS(sizeOfExpr(expr()))).bind(BinOp)
*/		    
		    binaryOperator(hasAnyOperatorName("+=","-=","+","-" ),
	              hasLHS(hasType(pointerType(pointee(qualType().bind(PointedType))))),
		      hasRHS(sizeOfExpr(expr()))).bind(BinOp),
		    binaryOperator(hasAnyOperatorName("+","-" ),
	              hasRHS(hasType(pointerType(pointee(qualType().bind(PointedType))))),
		      hasLHS(sizeOfExpr(expr()))).bind(BinOp)
            )),
        this);
}

static CharUnits getSizeOfType(const ASTContext &Ctx, const Type *Ty) {
  if (!Ty || Ty->isIncompleteType() || Ty->isDependentType() ||
      isa<DependentSizedArrayType>(Ty) || !Ty->isConstantSizeType())
    return CharUnits::Zero();
  return Ctx.getTypeSizeInChars(Ty);
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::check(const MatchFinder::MatchResult &Result) {
    const ASTContext &Ctx = *Result.Context;
    const auto *Matched = Result.Nodes.getNodeAs<BinaryOperator>(BinOp);
    const auto *SuspiciousQualTypePtr =
        Result.Nodes.getNodeAs<QualType>(PointedType);
    const auto *SuspiciousTypePtr = SuspiciousQualTypePtr->getTypePtr();

    std::size_t sz = getSizeOfType(Ctx,SuspiciousTypePtr).getQuantity();
    if ( sz > 1 )
    {
        diag(Matched->getExprLoc(),"Suspicious pointer arithmetics using sizeof() operator: sizeof(%0) is %1") << SuspiciousQualTypePtr->getAsString(Ctx.getPrintingPolicy())
	                       << sz
		               << Matched->getSourceRange();
    }
}

} // namespace clang::tidy::bugprone
