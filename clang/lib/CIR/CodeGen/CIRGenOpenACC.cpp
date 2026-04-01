//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic OpenACC lowering functions not Stmt, Decl, or clause specific.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
aiir::Value createBound(CIRGenFunction &cgf, CIRGen::CIRGenBuilderTy &builder,
                        aiir::Location boundLoc, aiir::Value lowerBound,
                        aiir::Value upperBound, aiir::Value extent) {
  // Arrays always have a start-idx of 0.
  aiir::Value startIdx = cgf.createOpenACCConstantInt(boundLoc, 64, 0);
  // Stride is always 1 in C/C++.
  aiir::Value stride = cgf.createOpenACCConstantInt(boundLoc, 64, 1);

  auto bound = aiir::acc::DataBoundsOp::create(builder, boundLoc, lowerBound,
                                               upperBound);
  bound.getStartIdxMutable().assign(startIdx);
  if (extent)
    bound.getExtentMutable().assign(extent);
  bound.getStrideMutable().assign(stride);

  return bound;
}
} // namespace

aiir::Value CIRGenFunction::emitOpenACCIntExpr(const Expr *intExpr) {
  aiir::Value expr = emitScalarExpr(intExpr);
  aiir::Location exprLoc = cgm.getLoc(intExpr->getBeginLoc());

  aiir::IntegerType targetType = aiir::IntegerType::get(
      &getAIIRContext(), getContext().getIntWidth(intExpr->getType()),
      intExpr->getType()->isSignedIntegerOrEnumerationType()
          ? aiir::IntegerType::SignednessSemantics::Signed
          : aiir::IntegerType::SignednessSemantics::Unsigned);

  auto conversionOp = aiir::UnrealizedConversionCastOp::create(
      builder, exprLoc, targetType, expr);
  return conversionOp.getResult(0);
}

aiir::Value CIRGenFunction::createOpenACCConstantInt(aiir::Location loc,
                                                     unsigned width,
                                                     int64_t value) {
  aiir::IntegerType ty =
      aiir::IntegerType::get(&getAIIRContext(), width,
                             aiir::IntegerType::SignednessSemantics::Signless);
  auto constOp = aiir::arith::ConstantOp::create(
      builder, loc, builder.getIntegerAttr(ty, value));

  return constOp;
}

CIRGenFunction::OpenACCDataOperandInfo
CIRGenFunction::getOpenACCDataOperandInfo(const Expr *e) {
  const Expr *curVarExpr = e->IgnoreParenImpCasts();
  QualType origType =
      curVarExpr->getType().getNonReferenceType().getUnqualifiedType();
  // Array sections are special, and we have to treat them that way.
  if (const auto *section =
          dyn_cast<ArraySectionExpr>(curVarExpr->IgnoreParenImpCasts()))
    origType = section->getElementType();

  aiir::Location exprLoc = cgm.getLoc(curVarExpr->getBeginLoc());
  llvm::SmallVector<aiir::Value> bounds;
  llvm::SmallVector<QualType> boundTypes;

  std::string exprString;
  llvm::raw_string_ostream os(exprString);
  e->printPretty(os, nullptr, getContext().getPrintingPolicy());

  auto addBoundType = [&](const Expr *e) {
    if (const auto *section = dyn_cast<ArraySectionExpr>(curVarExpr))
      boundTypes.push_back(section->getElementType());
    else
      boundTypes.push_back(curVarExpr->getType());
  };

  addBoundType(curVarExpr);

  while (isa<ArraySectionExpr, ArraySubscriptExpr>(curVarExpr)) {
    aiir::Location boundLoc = cgm.getLoc(curVarExpr->getBeginLoc());
    aiir::Value lowerBound;
    aiir::Value upperBound;
    aiir::Value extent;

    if (const auto *section = dyn_cast<ArraySectionExpr>(curVarExpr)) {
      if (const Expr *lb = section->getLowerBound())
        lowerBound = emitOpenACCIntExpr(lb);
      else
        lowerBound = createOpenACCConstantInt(boundLoc, 64, 0);

      if (const Expr *len = section->getLength()) {
        extent = emitOpenACCIntExpr(len);
      } else {
        QualType baseTy = section->getBaseType();
        // We know this is the case as implicit lengths are only allowed for
        // array types with a constant size, or a dependent size.  AND since
        // we are codegen we know we're not dependent.
        auto *arrayTy = getContext().getAsConstantArrayType(baseTy);
        // Rather than trying to calculate the extent based on the
        // lower-bound, we can just emit this as an upper bound.
        upperBound = createOpenACCConstantInt(boundLoc, 64,
                                              arrayTy->getLimitedSize() - 1);
      }

      curVarExpr = section->getBase()->IgnoreParenImpCasts();
    } else {
      const auto *subscript = cast<ArraySubscriptExpr>(curVarExpr);

      lowerBound = emitOpenACCIntExpr(subscript->getIdx());
      // Length of an array index is always 1.
      extent = createOpenACCConstantInt(boundLoc, 64, 1);
      curVarExpr = subscript->getBase()->IgnoreParenImpCasts();
    }

    bounds.push_back(createBound(*this, this->builder, boundLoc, lowerBound,
                                 upperBound, extent));
    addBoundType(curVarExpr);
  }

  if (const auto *memExpr = dyn_cast<MemberExpr>(curVarExpr))
    return {exprLoc,
            emitMemberExpr(memExpr).getPointer(),
            exprString,
            origType,
            curVarExpr->getType().getNonReferenceType().getUnqualifiedType(),
            std::move(bounds),
            std::move(boundTypes)};

  // Sema has made sure that only 4 types of things can get here, array
  // subscript, array section, member expr, or DRE to a var decl (or the
  // former 3 wrapping a var-decl), so we should be able to assume this is
  // right.
  const auto *dre = cast<DeclRefExpr>(curVarExpr);
  return {exprLoc,
          emitDeclRefLValue(dre).getPointer(),
          exprString,
          origType,
          curVarExpr->getType().getNonReferenceType().getUnqualifiedType(),
          std::move(bounds),
          std::move(boundTypes)};
}
