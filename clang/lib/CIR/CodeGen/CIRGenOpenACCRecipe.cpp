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

#include <numeric>

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
void OpenACCRecipeBuilderBase::makeAllocaCopy(mlir::Location loc,
                                              mlir::Type copyType,
                                              mlir::Value numEltsToCopy,
                                              mlir::Value offsetPerSubarray,
                                              mlir::Value destAlloca,
                                              mlir::Value srcAlloca) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);

  mlir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto itrPtrTy = cir::PointerType::get(itrTy);
  mlir::IntegerAttr itrAlign =
      cgf.cgm.getSize(cgf.getContext().getTypeAlignInChars(
          cgf.getContext().UnsignedLongLongTy));

  auto loopBuilder = [&]() {
    auto itr =
        cir::AllocaOp::create(builder, loc, itrPtrTy, itrTy, "itr", itrAlign);
    cir::ConstantOp constZero = builder.getConstInt(loc, itrTy, 0);
    builder.CIRBaseBuilderTy::createStore(loc, constZero, itr);
    builder.createFor(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // itr < numEltsToCopy
          // Enforce a trip count of 1 if there wasn't any element count, this
          // way we can just use this loop with a constant bounds instead of a
          // separate code path.
          if (!numEltsToCopy)
            numEltsToCopy = builder.getConstInt(loc, itrTy, 1);

          auto loadCur = cir::LoadOp::create(builder, loc, {itr});
          auto cmp = builder.createCompare(loc, cir::CmpOpKind::lt, loadCur,
                                           numEltsToCopy);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // destAlloca[itr] = srcAlloca[offsetPerSubArray * itr];
          auto loadCur = cir::LoadOp::create(builder, loc, {itr});
          auto srcOffset = builder.createMul(loc, offsetPerSubarray, loadCur);

          auto ptrToOffsetIntoSrc = cir::PtrStrideOp::create(
              builder, loc, copyType, srcAlloca, srcOffset);

          auto offsetIntoDecayDest = cir::PtrStrideOp::create(
              builder, loc, builder.getPointerTo(copyType), destAlloca,
              loadCur);

          builder.CIRBaseBuilderTy::createStore(loc, ptrToOffsetIntoSrc,
                                                offsetIntoDecayDest);
          builder.createYield(loc);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // Simple increment of the iterator.
          auto load = cir::LoadOp::create(builder, loc, {itr});
          auto inc = cir::UnaryOp::create(builder, loc, load.getType(),
                                          cir::UnaryOpKind::Inc, load);
          builder.CIRBaseBuilderTy::createStore(loc, inc, itr);
          builder.createYield(loc);
        });
  };

  cir::ScopeOp::create(builder, loc,
                       [&](mlir::OpBuilder &b, mlir::Location loc) {
                         loopBuilder();
                         builder.createYield(loc);
                       });
}

mlir::Value OpenACCRecipeBuilderBase::makeBoundsAlloca(
    mlir::Block *block, SourceRange exprRange, mlir::Location loc,
    std::string_view allocaName, size_t numBounds,
    llvm::ArrayRef<QualType> boundTypes) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);

  // Get the range of bounds arguments, which are all but the 1st arg.
  llvm::ArrayRef<mlir::BlockArgument> boundsRange =
      block->getArguments().drop_front(1);

  // boundTypes contains the before and after of each bounds, so it ends up
  // having 1 extra. Assert this is the case to ensure we don't call this in the
  // wrong 'block'.
  assert(boundsRange.size() + 1 == boundTypes.size());

  mlir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto idxType = mlir::IndexType::get(&cgf.getMLIRContext());

  auto getUpperBound = [&](mlir::Value bound) {
    auto upperBoundVal =
        mlir::acc::GetUpperboundOp::create(builder, loc, idxType, bound);
    return mlir::UnrealizedConversionCastOp::create(builder, loc, itrTy,
                                                    upperBoundVal.getResult())
        .getResult(0);
  };

  auto isArrayTy = [&](QualType ty) {
    if (ty->isArrayType() && !ty->isConstantArrayType())
      cgf.cgm.errorNYI(exprRange, "OpenACC recipe init for VLAs");
    return ty->isConstantArrayType();
  };

  mlir::Type topLevelTy = cgf.convertType(boundTypes.back());
  cir::PointerType topLevelTyPtr = builder.getPointerTo(topLevelTy);
  // Do an alloca for the 'top' level type without bounds.
  mlir::Value initialAlloca = builder.createAlloca(
      loc, topLevelTyPtr, topLevelTy, allocaName,
      cgf.getContext().getTypeAlignInChars(boundTypes.back()));

  bool lastBoundWasArray = isArrayTy(boundTypes.back());

  // Make sure we track a moving version of this so we can get our
  // 'copying' back to correct.
  mlir::Value lastAlloca = initialAlloca;

  // Since we're iterating the types in reverse, this sets up for each index
  // corresponding to the boundsRange to be the 'after application of the
  // bounds.
  llvm::ArrayRef<QualType> boundResults = boundTypes.drop_back(1);

  // Collect the 'do we have any allocas needed after this type' list.
  llvm::SmallVector<bool> allocasLeftArr;
  llvm::ArrayRef<QualType> resultTypes = boundTypes.drop_front();
  std::transform_inclusive_scan(
      resultTypes.begin(), resultTypes.end(),
      std::back_inserter(allocasLeftArr), std::plus<bool>{},
      [](QualType ty) { return !ty->isConstantArrayType(); }, false);

  // Keep track of the number of 'elements' that we're allocating. Individual
  // allocas should multiply this by the size of its current allocation.
  mlir::Value cumulativeElts;
  for (auto [bound, resultType, allocasLeft] : llvm::reverse(
           llvm::zip_equal(boundsRange, boundResults, allocasLeftArr))) {

    // if there is no further 'alloca' operation we need to do, we can skip
    // creating the UB/multiplications/etc.
    if (!allocasLeft)
      break;

    // First: figure out the number of elements in the current 'bound' list.
    mlir::Value eltsPerSubArray = getUpperBound(bound);
    mlir::Value eltsToAlloca;

    // IF we are in a sub-bounds, the total number of elements to alloca is
    // the product of that one and the current 'bounds' size.  That is,
    // arr[5][5], we would need 25 elements, not just 5. Else it is just the
    // current number of elements.
    if (cumulativeElts)
      eltsToAlloca = builder.createMul(loc, eltsPerSubArray, cumulativeElts);
    else
      eltsToAlloca = eltsPerSubArray;

    if (!lastBoundWasArray) {
      // If we have to do an allocation, figure out the size of the
      // allocation.  alloca takes the number of bytes, not elements.
      TypeInfoChars eltInfo = cgf.getContext().getTypeInfoInChars(resultType);
      cir::ConstantOp eltSize = builder.getConstInt(
          loc, itrTy, eltInfo.Width.alignTo(eltInfo.Align).getQuantity());
      mlir::Value curSize = builder.createMul(loc, eltsToAlloca, eltSize);

      mlir::Type eltTy = cgf.convertType(resultType);
      cir::PointerType ptrTy = builder.getPointerTo(eltTy);
      mlir::Value curAlloca = builder.createAlloca(
          loc, ptrTy, eltTy, "openacc.init.bounds",
          cgf.getContext().getTypeAlignInChars(resultType), curSize);

      makeAllocaCopy(loc, ptrTy, cumulativeElts, eltsPerSubArray, lastAlloca,
                     curAlloca);
      lastAlloca = curAlloca;
    } else {
      // In the case of an array, we just need to decay the pointer, so just do
      // a zero-offset stride on the last alloca to decay it down an array
      // level.
      cir::ConstantOp constZero = builder.getConstInt(loc, itrTy, 0);
      lastAlloca = builder.getArrayElement(loc, loc, lastAlloca,
                                           cgf.convertType(resultType),
                                           constZero, /*shouldDecay=*/true);
    }

    cumulativeElts = eltsToAlloca;
    lastBoundWasArray = isArrayTy(resultType);
  }
  return initialAlloca;
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
                                     idxLoad,
                                     /*shouldDecay=*/true);

    assert(isa<cir::PointerType>(eltTy));

    auto eltLoad = cir::LoadOp::create(builder, loc, {subVal});

    return cir::PtrStrideOp::create(builder, loc, eltLoad.getType(), eltLoad,
                                    idxLoad);
        
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

      auto sub = cir::BinOp::create(builder, loc, itrTy, cir::BinOpKind::Sub,
                                    ubConversion.getResult(0), constOne);

      // Upperbound is exclusive, so subtract 1.
      builder.CIRBaseBuilderTy::createStore(loc, sub, itr);
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
              loc, inverse ? cir::CmpOpKind::ge : cir::CmpOpKind::lt, loadCur,
              endItr.getResult(0));
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
          auto unary = cir::UnaryOp::create(
              builder, loc, load.getType(),
              inverse ? cir::UnaryOpKind::Dec : cir::UnaryOpKind::Inc, load);
          builder.CIRBaseBuilderTy::createStore(loc, unary, itr);
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
void OpenACCRecipeBuilderBase::makeBoundsInit(
    mlir::Value alloca, mlir::Location loc, mlir::Block *block,
    const VarDecl *allocaDecl, QualType origType, bool isInitSection) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(block);
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  CIRGenFunction::AutoVarEmission tempDeclEmission{*allocaDecl};
  tempDeclEmission.emittedAsOffload = true;

  // The init section is the only one of the handful that only has a single
  // argument for the 'type', so we have to drop 1 for init, and future calls
  // to this will need to drop 2.
  llvm::MutableArrayRef<mlir::BlockArgument> boundsRange =
      block->getArguments().drop_front(isInitSection ? 1 : 2);

  mlir::Value subscriptedValue = alloca;
  for (mlir::BlockArgument boundArg : llvm::reverse(boundsRange))
    subscriptedValue = createBoundsLoop(subscriptedValue, boundArg, loc,
                                        /*inverse=*/false);

  tempDeclEmission.setAllocatedAddress(
      Address{subscriptedValue, cgf.convertType(origType),
              cgf.getContext().getDeclAlign(allocaDecl)});
  cgf.emitAutoVarInit(tempDeclEmission);
}

// TODO: OpenACC: When we get this implemented for the reduction/firstprivate,
// this might end up re-merging with createRecipeInitCopy.  For now, keep it
// separate until we're sure what everything looks like to keep this as clean
// as possible.
void OpenACCRecipeBuilderBase::createPrivateInitRecipe(
    mlir::Location loc, mlir::Location locEnd, SourceRange exprRange,
    mlir::Value mainOp, mlir::acc::PrivateRecipeOp recipe, size_t numBounds,
    llvm::ArrayRef<QualType> boundTypes, const VarDecl *allocaDecl,
    QualType origType) {
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
    mlir::Value alloca = makeBoundsAlloca(
        block, exprRange, loc, "openacc.private.init", numBounds, boundTypes);

    // If the initializer is trivial, there is nothing to do here, so save
    // ourselves some effort.
    if (allocaDecl->getInit() &&
        (!cgf.isTrivialInitializer(allocaDecl->getInit()) ||
         cgf.getContext().getLangOpts().getTrivialAutoVarInit() !=
             LangOptions::TrivialAutoVarInitKind::Uninitialized))
      makeBoundsInit(alloca, loc, block, allocaDecl, origType,
                     /*isInitSection=*/true);
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
  tempDeclEmission.emittedAsOffload = true;

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
