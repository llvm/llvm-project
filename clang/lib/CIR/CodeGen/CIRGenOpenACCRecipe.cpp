//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helperes to emit OpenACC clause recipes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenACCRecipe.h"

namespace clang::CIRGen {
mlir::Block *OpenACCRecipeBuilderBase::createRecipeBlock(mlir::Region &region,
                                                         mlir::Type opTy,
                                                         mlir::Location loc,
                                                         size_t numBounds,
                                                         bool isInit) {
  llvm::SmallVector<mlir::Type> types;
  types.reserve(numBounds + 2);
  types.push_back(opTy);
  // The init section is the only one that doesn't have TWO copies of the
  // operation-type.  Copy has a to/from, and destroy has a
  // 'reference'/'privatized' copy version.
  if (!isInit)
    types.push_back(opTy);

  auto boundsTy = mlir::acc::DataBoundsType::get(&cgf.getMLIRContext());
  for (size_t i = 0; i < numBounds; ++i)
    types.push_back(boundsTy);

  llvm::SmallVector<mlir::Location> locs{types.size(), loc};
  return builder.createBlock(&region, region.end(), types, locs);
}

mlir::Value
OpenACCRecipeBuilderBase::createBoundsLoop(mlir::Value subscriptedValue,
                                           mlir::Value bound,
                                           mlir::Location loc, bool inverse) {
  mlir::Operation *bodyInsertLoc;

  mlir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto itrPtrTy = cir::PointerType::get(itrTy);
  mlir::IntegerAttr itrAlign =
      cgf.cgm.getSize(cgf.getContext().getTypeAlignInChars(
          cgf.getContext().UnsignedLongLongTy));
  auto idxType = mlir::IndexType::get(&cgf.getMLIRContext());

  auto doSubscriptOp = [&](mlir::Value subVal,
                           cir::LoadOp idxLoad) -> mlir::Value {
    auto eltTy = cast<cir::PointerType>(subVal.getType()).getPointee();

    if (auto arrayTy = dyn_cast<cir::ArrayType>(eltTy))
      return builder.getArrayElement(loc, loc, subVal, arrayTy.getElementType(),
                                     idxLoad.getResult(),
                                     /*shouldDecay=*/true);

    assert(isa<cir::PointerType>(eltTy));

    auto eltLoad = cir::LoadOp::create(builder, loc, {subVal});

    return cir::PtrStrideOp::create(builder, loc, eltLoad.getType(), eltLoad,
                                    idxLoad.getResult())
        .getResult();
  };

  auto forStmtBuilder = [&]() {
    // get the lower and upper bound for iterating over.
    auto lowerBoundVal =
        mlir::acc::GetLowerboundOp::create(builder, loc, idxType, bound);
    auto lbConversion = mlir::UnrealizedConversionCastOp::create(
        builder, loc, itrTy, lowerBoundVal.getResult());
    auto upperBoundVal =
        mlir::acc::GetUpperboundOp::create(builder, loc, idxType, bound);
    auto ubConversion = mlir::UnrealizedConversionCastOp::create(
        builder, loc, itrTy, upperBoundVal.getResult());

    // Create a memory location for the iterator.
    auto itr =
        cir::AllocaOp::create(builder, loc, itrPtrTy, itrTy, "iter", itrAlign);
    // Store to the iterator: either lower bound, or if inverse loop, upper
    // bound.
    if (inverse) {
      cir::ConstantOp constOne = builder.getConstInt(loc, itrTy, 1);

      auto sub =
          cir::BinOp::create(builder, loc, itrTy, cir::BinOpKind::Sub,
                             ubConversion.getResult(0), constOne.getResult());

      // Upperbound is exclusive, so subtract 1.
      builder.CIRBaseBuilderTy::createStore(loc, sub.getResult(), itr);
    } else {
      // Lowerbound is inclusive, so we can include it.
      builder.CIRBaseBuilderTy::createStore(loc, lbConversion.getResult(0),
                                            itr);
    }
    // Save the 'end' iterator based on whether we are inverted or not. This
    // end iterator never changes, so we can just get it and convert it, so no
    // need to store/load/etc.
    auto endItr = inverse ? lbConversion : ubConversion;

    builder.createFor(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto loadCur = cir::LoadOp::create(builder, loc, {itr});
          // Use 'not equal' since we are just doing an increment/decrement.
          auto cmp = builder.createCompare(
              loc, inverse ? cir::CmpOpKind::ge : cir::CmpOpKind::lt,
              loadCur.getResult(), endItr.getResult(0));
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto load = cir::LoadOp::create(builder, loc, {itr});

          if (subscriptedValue)
            subscriptedValue = doSubscriptOp(subscriptedValue, load);
          bodyInsertLoc = builder.createYield(loc);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto load = cir::LoadOp::create(builder, loc, {itr});
          auto unary = cir::UnaryOp::create(builder, loc, load.getType(),
                                            inverse ? cir::UnaryOpKind::Dec
                                                    : cir::UnaryOpKind::Inc,
                                            load.getResult());
          builder.CIRBaseBuilderTy::createStore(loc, unary.getResult(), itr);
          builder.createYield(loc);
        });
  };

  cir::ScopeOp::create(builder, loc,
                       [&](mlir::OpBuilder &b, mlir::Location loc) {
                         forStmtBuilder();
                         builder.createYield(loc);
                       });

  // Leave the insertion point to be inside the body, so we can loop over
  // these things.
  builder.setInsertionPoint(bodyInsertLoc);
  return subscriptedValue;
}

mlir::acc::ReductionOperator
OpenACCRecipeBuilderBase::convertReductionOp(OpenACCReductionOperator op) {
  switch (op) {
  case OpenACCReductionOperator::Addition:
    return mlir::acc::ReductionOperator::AccAdd;
  case OpenACCReductionOperator::Multiplication:
    return mlir::acc::ReductionOperator::AccMul;
  case OpenACCReductionOperator::Max:
    return mlir::acc::ReductionOperator::AccMax;
  case OpenACCReductionOperator::Min:
    return mlir::acc::ReductionOperator::AccMin;
  case OpenACCReductionOperator::BitwiseAnd:
    return mlir::acc::ReductionOperator::AccIand;
  case OpenACCReductionOperator::BitwiseOr:
    return mlir::acc::ReductionOperator::AccIor;
  case OpenACCReductionOperator::BitwiseXOr:
    return mlir::acc::ReductionOperator::AccXor;
  case OpenACCReductionOperator::And:
    return mlir::acc::ReductionOperator::AccLand;
  case OpenACCReductionOperator::Or:
    return mlir::acc::ReductionOperator::AccLor;
  case OpenACCReductionOperator::Invalid:
    llvm_unreachable("invalid reduction operator");
  }

  llvm_unreachable("invalid reduction operator");
}

// This function generates the 'destroy' section for a recipe. Note
// that this function is not 'insertion point' clean, in that it alters the
// insertion point to be inside of the 'destroy' section of the recipe, but
// doesn't restore it aftewards.
void OpenACCRecipeBuilderBase::createRecipeDestroySection(
    mlir::Location loc, mlir::Location locEnd, mlir::Value mainOp,
    CharUnits alignment, QualType origType, size_t numBounds, QualType baseType,
    mlir::Region &destroyRegion) {
  mlir::Block *block = createRecipeBlock(destroyRegion, mainOp.getType(), loc,
                                         numBounds, /*isInit=*/false);
  builder.setInsertionPointToEnd(&destroyRegion.back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  mlir::Type elementTy =
      mlir::cast<cir::PointerType>(mainOp.getType()).getPointee();
  auto emitDestroy = [&](mlir::Value var, mlir::Type ty) {
    Address addr{var, ty, alignment};
    cgf.emitDestroy(addr, origType,
                    cgf.getDestroyer(QualType::DK_cxx_destructor));
  };

  if (numBounds) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    // Get the range of bounds arguments, which are all but the 1st 2. 1st is
    // a 'reference', 2nd is the 'private' variant we need to destroy from.
    llvm::MutableArrayRef<mlir::BlockArgument> boundsRange =
        block->getArguments().drop_front(2);

    mlir::Value subscriptedValue = block->getArgument(1);
    for (mlir::BlockArgument boundArg : llvm::reverse(boundsRange))
      subscriptedValue = createBoundsLoop(subscriptedValue, boundArg, loc,
                                          /*inverse=*/true);

    emitDestroy(subscriptedValue, cgf.cgm.convertType(origType));
  } else {
    // If we don't have any bounds, we can just destroy the variable directly.
    // The destroy region has a signature of "original item, privatized item".
    // So the 2nd item is the one that needs destroying, the former is just
    // for reference and we don't really have a need for it at the moment.
    emitDestroy(block->getArgument(1), elementTy);
  }

  mlir::acc::YieldOp::create(builder, locEnd);
}

// TODO: OpenACC: When we get this implemented for the reduction/firstprivate,
// this might end up re-merging with createRecipeInitCopy.  For now, keep it
// separate until we're sure what everything looks like to keep this as clean
// as possible.
void OpenACCRecipeBuilderBase::createPrivateInitRecipe(
    mlir::Location loc, mlir::Location locEnd, SourceRange exprRange,
    mlir::Value mainOp, mlir::acc::PrivateRecipeOp recipe, size_t numBounds,
    llvm::ArrayRef<QualType> boundTypes, const VarDecl *allocaDecl,
    QualType origType, const Expr *initExpr) {
  assert(allocaDecl && "Required recipe variable not set?");
  CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, allocaDecl};

  mlir::Block *block =
      createRecipeBlock(recipe.getInitRegion(), mainOp.getType(), loc,
                        numBounds, /*isInit=*/true);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  const Type *allocaPointeeType =
      allocaDecl->getType()->getPointeeOrArrayElementType();
  // We are OK with no init for builtins, arrays of builtins, or pointers,
  // else we should NYI so we know to go look for these.
  if (cgf.getContext().getLangOpts().CPlusPlus && !allocaDecl->getInit() &&
      !allocaDecl->getType()->isPointerType() &&
      !allocaPointeeType->isBuiltinType() &&
      !allocaPointeeType->isPointerType()) {
    // If we don't have any initialization recipe, we failed during Sema to
    // initialize this correctly. If we disable the
    // Sema::TentativeAnalysisScopes in SemaOpenACC::CreateInitRecipe, it'll
    // emit an error to tell us.  However, emitting those errors during
    // production is a violation of the standard, so we cannot do them.
    cgf.cgm.errorNYI(exprRange, "private default-init recipe");
  }

  if (!numBounds) {
    // This is an 'easy' case, we just have to use the builtin init stuff to
    // initialize this variable correctly.
    CIRGenFunction::AutoVarEmission tempDeclEmission =
        cgf.emitAutoVarAlloca(*allocaDecl, builder.saveInsertionPoint());
    cgf.emitAutoVarInit(tempDeclEmission);
  } else {
    cgf.cgm.errorNYI(exprRange, "private-init with bounds");
  }

  mlir::acc::YieldOp::create(builder, locEnd);
}

void OpenACCRecipeBuilderBase::createFirstprivateRecipeCopy(
    mlir::Location loc, mlir::Location locEnd, mlir::Value mainOp,
    CIRGenFunction::AutoVarEmission tempDeclEmission,
    mlir::acc::FirstprivateRecipeOp recipe, const VarDecl *varRecipe,
    const VarDecl *temporary) {
  mlir::Block *block =
      createRecipeBlock(recipe.getCopyRegion(), mainOp.getType(), loc,
                        /*numBounds=*/0, /*isInit=*/false);
  builder.setInsertionPointToEnd(&recipe.getCopyRegion().back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  mlir::BlockArgument fromArg = block->getArgument(0);
  mlir::BlockArgument toArg = block->getArgument(1);

  mlir::Type elementTy =
      mlir::cast<cir::PointerType>(mainOp.getType()).getPointee();

  // Set the address of the emission to be the argument, so that we initialize
  // that instead of the variable in the other block.
  tempDeclEmission.setAllocatedAddress(
      Address{toArg, elementTy, cgf.getContext().getDeclAlign(varRecipe)});
  tempDeclEmission.EmittedAsOffload = true;

  CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, temporary};
  cgf.setAddrOfLocalVar(
      temporary,
      Address{fromArg, elementTy, cgf.getContext().getDeclAlign(varRecipe)});

  cgf.emitAutoVarInit(tempDeclEmission);
  mlir::acc::YieldOp::create(builder, locEnd);
}
// This function generates the 'combiner' section for a reduction recipe. Note
// that this function is not 'insertion point' clean, in that it alters the
// insertion point to be inside of the 'combiner' section of the recipe, but
// doesn't restore it aftewards.
void OpenACCRecipeBuilderBase::createReductionRecipeCombiner(
    mlir::Location loc, mlir::Location locEnd, mlir::Value mainOp,
    mlir::acc::ReductionRecipeOp recipe) {
  mlir::Block *block = builder.createBlock(
      &recipe.getCombinerRegion(), recipe.getCombinerRegion().end(),
      {mainOp.getType(), mainOp.getType()}, {loc, loc});
  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  mlir::BlockArgument lhsArg = block->getArgument(0);

  mlir::acc::YieldOp::create(builder, locEnd, lhsArg);
}

} // namespace clang::CIRGen
