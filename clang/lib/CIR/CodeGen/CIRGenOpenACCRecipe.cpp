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
aiir::Block *OpenACCRecipeBuilderBase::createRecipeBlock(aiir::Region &region,
                                                         aiir::Type opTy,
                                                         aiir::Location loc,
                                                         size_t numBounds,
                                                         bool isInit) {
  llvm::SmallVector<aiir::Type> types;
  types.reserve(numBounds + 2);
  types.push_back(opTy);
  // The init section is the only one that doesn't have TWO copies of the
  // operation-type.  Copy has a to/from, and destroy has a
  // 'reference'/'privatized' copy version.
  if (!isInit)
    types.push_back(opTy);

  auto boundsTy = aiir::acc::DataBoundsType::get(&cgf.getAIIRContext());
  for (size_t i = 0; i < numBounds; ++i)
    types.push_back(boundsTy);

  llvm::SmallVector<aiir::Location> locs{types.size(), loc};
  return builder.createBlock(&region, region.end(), types, locs);
}
void OpenACCRecipeBuilderBase::makeAllocaCopy(aiir::Location loc,
                                              aiir::Type copyType,
                                              aiir::Value numEltsToCopy,
                                              aiir::Value offsetPerSubarray,
                                              aiir::Value destAlloca,
                                              aiir::Value srcAlloca) {
  aiir::OpBuilder::InsertionGuard guardCase(builder);

  aiir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto itrPtrTy = cir::PointerType::get(itrTy);
  aiir::IntegerAttr itrAlign =
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
        [&](aiir::OpBuilder &b, aiir::Location loc) {
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
        [&](aiir::OpBuilder &b, aiir::Location loc) {
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
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          // Simple increment of the iterator.
          auto load = cir::LoadOp::create(builder, loc, {itr});
          auto inc = builder.createInc(loc, load);
          builder.CIRBaseBuilderTy::createStore(loc, inc, itr);
          builder.createYield(loc);
        });
  };

  cir::ScopeOp::create(builder, loc,
                       [&](aiir::OpBuilder &b, aiir::Location loc) {
                         loopBuilder();
                         builder.createYield(loc);
                       });
}

aiir::Value OpenACCRecipeBuilderBase::makeBoundsAlloca(
    aiir::Block *block, SourceRange exprRange, aiir::Location loc,
    std::string_view allocaName, size_t numBounds,
    llvm::ArrayRef<QualType> boundTypes) {
  aiir::OpBuilder::InsertionGuard guardCase(builder);

  // Get the range of bounds arguments, which are all but the 1st arg.
  llvm::ArrayRef<aiir::BlockArgument> boundsRange =
      block->getArguments().drop_front(1);

  // boundTypes contains the before and after of each bounds, so it ends up
  // having 1 extra. Assert this is the case to ensure we don't call this in the
  // wrong 'block'.
  assert(boundsRange.size() + 1 == boundTypes.size());

  aiir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto idxType = aiir::IndexType::get(&cgf.getAIIRContext());

  auto getUpperBound = [&](aiir::Value bound) {
    auto upperBoundVal =
        aiir::acc::GetUpperboundOp::create(builder, loc, idxType, bound);
    return aiir::UnrealizedConversionCastOp::create(builder, loc, itrTy,
                                                    upperBoundVal.getResult())
        .getResult(0);
  };

  auto isArrayTy = [&](QualType ty) {
    if (ty->isArrayType() && !ty->isConstantArrayType())
      cgf.cgm.errorNYI(exprRange, "OpenACC recipe init for VLAs");
    return ty->isConstantArrayType();
  };

  aiir::Type topLevelTy = cgf.convertType(boundTypes.back());
  cir::PointerType topLevelTyPtr = builder.getPointerTo(topLevelTy);
  // Do an alloca for the 'top' level type without bounds.
  aiir::Value initialAlloca = builder.createAlloca(
      loc, topLevelTyPtr, topLevelTy, allocaName,
      cgf.getContext().getTypeAlignInChars(boundTypes.back()));

  bool lastBoundWasArray = isArrayTy(boundTypes.back());

  // Make sure we track a moving version of this so we can get our
  // 'copying' back to correct.
  aiir::Value lastAlloca = initialAlloca;

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
  aiir::Value cumulativeElts;
  for (auto [bound, resultType, allocasLeft] : llvm::reverse(
           llvm::zip_equal(boundsRange, boundResults, allocasLeftArr))) {

    // if there is no further 'alloca' operation we need to do, we can skip
    // creating the UB/multiplications/etc.
    if (!allocasLeft)
      break;

    // First: figure out the number of elements in the current 'bound' list.
    aiir::Value eltsPerSubArray = getUpperBound(bound);
    aiir::Value eltsToAlloca;

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
      aiir::Value curSize = builder.createMul(loc, eltsToAlloca, eltSize);

      aiir::Type eltTy = cgf.convertType(resultType);
      cir::PointerType ptrTy = builder.getPointerTo(eltTy);
      aiir::Value curAlloca = builder.createAlloca(
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

std::pair<aiir::Value, aiir::Value> OpenACCRecipeBuilderBase::createBoundsLoop(
    aiir::Value subscriptedValue, aiir::Value subscriptedValue2,
    aiir::Value bound, aiir::Location loc, bool inverse) {
  aiir::Operation *bodyInsertLoc;

  aiir::Type itrTy = cgf.cgm.convertType(cgf.getContext().UnsignedLongLongTy);
  auto itrPtrTy = cir::PointerType::get(itrTy);
  aiir::IntegerAttr itrAlign =
      cgf.cgm.getSize(cgf.getContext().getTypeAlignInChars(
          cgf.getContext().UnsignedLongLongTy));
  auto idxType = aiir::IndexType::get(&cgf.getAIIRContext());

  auto doSubscriptOp = [&](aiir::Value subVal,
                           cir::LoadOp idxLoad) -> aiir::Value {
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
        aiir::acc::GetLowerboundOp::create(builder, loc, idxType, bound);
    auto lbConversion = aiir::UnrealizedConversionCastOp::create(
        builder, loc, itrTy, lowerBoundVal.getResult());
    auto upperBoundVal =
        aiir::acc::GetUpperboundOp::create(builder, loc, idxType, bound);
    auto ubConversion = aiir::UnrealizedConversionCastOp::create(
        builder, loc, itrTy, upperBoundVal.getResult());

    // Create a memory location for the iterator.
    auto itr =
        cir::AllocaOp::create(builder, loc, itrPtrTy, itrTy, "iter", itrAlign);
    // Store to the iterator: either lower bound, or if inverse loop, upper
    // bound.
    if (inverse) {
      cir::ConstantOp constOne = builder.getConstInt(loc, itrTy, 1);

      auto sub =
          cir::SubOp::create(builder, loc, ubConversion.getResult(0), constOne);

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
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto loadCur = cir::LoadOp::create(builder, loc, {itr});
          // Use 'not equal' since we are just doing an increment/decrement.
          auto cmp = builder.createCompare(
              loc, inverse ? cir::CmpOpKind::ge : cir::CmpOpKind::lt, loadCur,
              endItr.getResult(0));
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto load = cir::LoadOp::create(builder, loc, {itr});

          if (subscriptedValue)
            subscriptedValue = doSubscriptOp(subscriptedValue, load);
          if (subscriptedValue2)
            subscriptedValue2 = doSubscriptOp(subscriptedValue2, load);
          bodyInsertLoc = builder.createYield(loc);
        },
        /*stepBuilder=*/
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto load = cir::LoadOp::create(builder, loc, {itr});
          auto unary = inverse ? builder.createDec(loc, load)
                               : builder.createInc(loc, load);
          builder.CIRBaseBuilderTy::createStore(loc, unary, itr);
          builder.createYield(loc);
        });
  };

  cir::ScopeOp::create(builder, loc,
                       [&](aiir::OpBuilder &b, aiir::Location loc) {
                         forStmtBuilder();
                         builder.createYield(loc);
                       });

  // Leave the insertion point to be inside the body, so we can loop over
  // these things.
  builder.setInsertionPoint(bodyInsertLoc);
  return {subscriptedValue, subscriptedValue2};
}

aiir::acc::ReductionOperator
OpenACCRecipeBuilderBase::convertReductionOp(OpenACCReductionOperator op) {
  switch (op) {
  case OpenACCReductionOperator::Addition:
    return aiir::acc::ReductionOperator::AccAdd;
  case OpenACCReductionOperator::Multiplication:
    return aiir::acc::ReductionOperator::AccMul;
  case OpenACCReductionOperator::Max:
    return aiir::acc::ReductionOperator::AccMax;
  case OpenACCReductionOperator::Min:
    return aiir::acc::ReductionOperator::AccMin;
  case OpenACCReductionOperator::BitwiseAnd:
    return aiir::acc::ReductionOperator::AccIand;
  case OpenACCReductionOperator::BitwiseOr:
    return aiir::acc::ReductionOperator::AccIor;
  case OpenACCReductionOperator::BitwiseXOr:
    return aiir::acc::ReductionOperator::AccXor;
  case OpenACCReductionOperator::And:
    return aiir::acc::ReductionOperator::AccLand;
  case OpenACCReductionOperator::Or:
    return aiir::acc::ReductionOperator::AccLor;
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
    aiir::Location loc, aiir::Location locEnd, aiir::Value mainOp,
    CharUnits alignment, QualType origType, size_t numBounds, QualType baseType,
    aiir::Region &destroyRegion) {
  aiir::Block *block = createRecipeBlock(destroyRegion, mainOp.getType(), loc,
                                         numBounds, /*isInit=*/false);
  builder.setInsertionPointToEnd(&destroyRegion.back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  aiir::Type elementTy =
      aiir::cast<cir::PointerType>(mainOp.getType()).getPointee();
  auto emitDestroy = [&](aiir::Value var, aiir::Type ty) {
    Address addr{var, ty, alignment};
    cgf.emitDestroy(addr, origType,
                    cgf.getDestroyer(QualType::DK_cxx_destructor));
  };

  if (numBounds) {
    aiir::OpBuilder::InsertionGuard guardCase(builder);
    // Get the range of bounds arguments, which are all but the 1st 2. 1st is
    // a 'reference', 2nd is the 'private' variant we need to destroy from.
    llvm::MutableArrayRef<aiir::BlockArgument> boundsRange =
        block->getArguments().drop_front(2);

    aiir::Value subscriptedValue = block->getArgument(1);
    for (aiir::BlockArgument boundArg : llvm::reverse(boundsRange))
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

  ls.forceCleanup();
  aiir::acc::YieldOp::create(builder, locEnd);
}
void OpenACCRecipeBuilderBase::makeBoundsInit(
    aiir::Value alloca, aiir::Location loc, aiir::Block *block,
    const VarDecl *allocaDecl, QualType origType, bool isInitSection) {
  aiir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(block);
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  CIRGenFunction::AutoVarEmission tempDeclEmission{*allocaDecl};
  tempDeclEmission.emittedAsOffload = true;

  // The init section is the only one of the handful that only has a single
  // argument for the 'type', so we have to drop 1 for init, and future calls
  // to this will need to drop 2.
  llvm::MutableArrayRef<aiir::BlockArgument> boundsRange =
      block->getArguments().drop_front(isInitSection ? 1 : 2);

  aiir::Value subscriptedValue = alloca;
  for (aiir::BlockArgument boundArg : llvm::reverse(boundsRange))
    subscriptedValue = createBoundsLoop(subscriptedValue, boundArg, loc,
                                        /*inverse=*/false);

  tempDeclEmission.setAllocatedAddress(
      Address{subscriptedValue, cgf.convertType(origType),
              cgf.getContext().getDeclAlign(allocaDecl)});
  cgf.emitAutoVarInit(tempDeclEmission);
}

// TODO: OpenACC: when we start doing firstprivate for array/vlas/etc, we
// probably need to do a little work about the 'init' calls to put it in 'copy'
// region instead.
void OpenACCRecipeBuilderBase::createInitRecipe(
    aiir::Location loc, aiir::Location locEnd, SourceRange exprRange,
    aiir::Value mainOp, aiir::Region &recipeInitRegion, size_t numBounds,
    llvm::ArrayRef<QualType> boundTypes, const VarDecl *allocaDecl,
    QualType origType, bool emitInitExpr) {
  assert(allocaDecl && "Required recipe variable not set?");
  CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, allocaDecl};

  aiir::Block *block = createRecipeBlock(recipeInitRegion, mainOp.getType(),
                                         loc, numBounds, /*isInit=*/true);
  builder.setInsertionPointToEnd(&recipeInitRegion.back());
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
    cgf.cgm.errorNYI(exprRange, "private/reduction default-init recipe");
  }

  if (!numBounds) {
    // This is an 'easy' case, we just have to use the builtin init stuff to
    // initialize this variable correctly.
    CIRGenFunction::AutoVarEmission tempDeclEmission =
        cgf.emitAutoVarAlloca(*allocaDecl, builder.saveInsertionPoint());
    if (emitInitExpr)
      cgf.emitAutoVarInit(tempDeclEmission);
  } else {
    aiir::Value alloca = makeBoundsAlloca(
        block, exprRange, loc, allocaDecl->getName(), numBounds, boundTypes);

    // If the initializer is trivial, there is nothing to do here, so save
    // ourselves some effort.
    if (emitInitExpr && allocaDecl->getInit() &&
        (!cgf.isTrivialInitializer(allocaDecl->getInit()) ||
         cgf.getContext().getLangOpts().getTrivialAutoVarInit() !=
             LangOptions::TrivialAutoVarInitKind::Uninitialized))
      makeBoundsInit(alloca, loc, block, allocaDecl, origType,
                     /*isInitSection=*/true);
  }

  ls.forceCleanup();
  aiir::acc::YieldOp::create(builder, locEnd);
}

void OpenACCRecipeBuilderBase::createFirstprivateRecipeCopy(
    aiir::Location loc, aiir::Location locEnd, aiir::Value mainOp,
    const VarDecl *allocaDecl, const VarDecl *temporary,
    aiir::Region &copyRegion, size_t numBounds) {
  aiir::Block *block = createRecipeBlock(copyRegion, mainOp.getType(), loc,
                                         numBounds, /*isInit=*/false);
  builder.setInsertionPointToEnd(&copyRegion.back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  aiir::Value fromArg = block->getArgument(0);
  aiir::Value toArg = block->getArgument(1);

  llvm::MutableArrayRef<aiir::BlockArgument> boundsRange =
      block->getArguments().drop_front(2);

  for (aiir::BlockArgument boundArg : llvm::reverse(boundsRange))
    std::tie(fromArg, toArg) =
        createBoundsLoop(fromArg, toArg, boundArg, loc, /*inverse=*/false);

  // Set up the 'to' address.
  aiir::Type elementTy =
      aiir::cast<cir::PointerType>(toArg.getType()).getPointee();
  CIRGenFunction::AutoVarEmission tempDeclEmission(*allocaDecl);
  tempDeclEmission.emittedAsOffload = true;
  tempDeclEmission.setAllocatedAddress(
      Address{toArg, elementTy, cgf.getContext().getDeclAlign(allocaDecl)});

  // Set up the 'from' address from the temporary.
  CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, temporary};
  cgf.setAddrOfLocalVar(
      temporary,
      Address{fromArg, elementTy, cgf.getContext().getDeclAlign(allocaDecl)});
  cgf.emitAutoVarInit(tempDeclEmission);

  builder.setInsertionPointToEnd(&copyRegion.back());
  ls.forceCleanup();
  aiir::acc::YieldOp::create(builder, locEnd);
}

// This function generates the 'combiner' section for a reduction recipe. Note
// that this function is not 'insertion point' clean, in that it alters the
// insertion point to be inside of the 'combiner' section of the recipe, but
// doesn't restore it aftewards.
void OpenACCRecipeBuilderBase::createReductionRecipeCombiner(
    aiir::Location loc, aiir::Location locEnd, aiir::Value mainOp,
    aiir::acc::ReductionRecipeOp recipe, size_t numBounds, QualType origType,
    llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe> combinerRecipes) {
  aiir::Block *block =
      createRecipeBlock(recipe.getCombinerRegion(), mainOp.getType(), loc,
                        numBounds, /*isInit=*/false);
  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  CIRGenFunction::LexicalScope ls(cgf, loc, block);

  aiir::Value lhsArg = block->getArgument(0);
  aiir::Value rhsArg = block->getArgument(1);
  llvm::MutableArrayRef<aiir::BlockArgument> boundsRange =
      block->getArguments().drop_front(2);

  if (llvm::any_of(combinerRecipes, [](auto &r) { return r.Op == nullptr; })) {
    cgf.cgm.errorNYI(loc, "OpenACC Reduction combiner not generated");
    aiir::acc::YieldOp::create(builder, locEnd, block->getArgument(0));
    return;
  }

  // apply the bounds so that we can get our bounds emitted correctly.
  for (aiir::BlockArgument boundArg : llvm::reverse(boundsRange))
    std::tie(lhsArg, rhsArg) =
        createBoundsLoop(lhsArg, rhsArg, boundArg, loc, /*inverse=*/false);

  // Emitter for when we know this isn't a struct or array we have to loop
  // through. This should work for the 'field' once the get-element call has
  // been made.
  auto emitSingleCombiner =
      [&](aiir::Value lhsArg, aiir::Value rhsArg,
          const OpenACCReductionRecipe::CombinerRecipe &combiner) {
        aiir::Type elementTy =
            aiir::cast<cir::PointerType>(lhsArg.getType()).getPointee();
        CIRGenFunction::DeclMapRevertingRAII declMapRAIILhs{cgf, combiner.LHS};
        cgf.setAddrOfLocalVar(
            combiner.LHS, Address{lhsArg, elementTy,
                                  cgf.getContext().getDeclAlign(combiner.LHS)});
        CIRGenFunction::DeclMapRevertingRAII declMapRAIIRhs{cgf, combiner.RHS};
        cgf.setAddrOfLocalVar(
            combiner.RHS, Address{rhsArg, elementTy,
                                  cgf.getContext().getDeclAlign(combiner.RHS)});

        [[maybe_unused]] aiir::LogicalResult stmtRes =
            cgf.emitStmt(combiner.Op, /*useCurrentScope=*/true);
      };

  // Emitter for when we know this is either a non-array or element of an array
  // (which also shouldn't be an array type?). This function should generate the
  // initialization code for an entire 'array-element'/non-array, including
  // diving into each element of a struct (if necessary).
  auto emitCombiner = [&](aiir::Value lhsArg, aiir::Value rhsArg, QualType ty) {
    assert(!ty->isArrayType() && "Array type shouldn't get here");
    if (const auto *rd = ty->getAsRecordDecl()) {
      if (combinerRecipes.size() == 1 &&
          cgf.getContext().hasSameType(ty, combinerRecipes[0].LHS->getType())) {
        // If this is a 'top level' operator on the type we can just emit this
        // as a simple one.
        emitSingleCombiner(lhsArg, rhsArg, combinerRecipes[0]);
      } else {
        // else we have to handle each individual field after after a
        // get-element.
        const CIRGenRecordLayout &layout =
            cgf.cgm.getTypes().getCIRGenRecordLayout(rd);
        for (const auto &[field, combiner] :
             llvm::zip_equal(rd->fields(), combinerRecipes)) {
          aiir::Type fieldType = cgf.convertType(field->getType());
          auto fieldPtr = cir::PointerType::get(fieldType);
          unsigned fieldIndex = layout.getCIRFieldNo(field);

          aiir::Value lhsField = builder.createGetMember(
              loc, fieldPtr, lhsArg, field->getName(), fieldIndex);
          aiir::Value rhsField = builder.createGetMember(
              loc, fieldPtr, rhsArg, field->getName(), fieldIndex);

          emitSingleCombiner(lhsField, rhsField, combiner);
        }
      }

    } else {
      // if this is a single-thing (because we should know this isn't an array,
      // as Sema wouldn't let us get here), we can just do a normal emit call.
      emitSingleCombiner(lhsArg, rhsArg, combinerRecipes[0]);
    }
  };

  if (const auto *cat = cgf.getContext().getAsConstantArrayType(origType)) {
    // If we're in an array, we have to emit the combiner for each element of
    // the array.
    auto itrTy = aiir::cast<cir::IntType>(cgf.ptrDiffTy);
    auto itrPtrTy = cir::PointerType::get(itrTy);

    aiir::Value zero =
        builder.getConstInt(loc, aiir::cast<cir::IntType>(cgf.ptrDiffTy), 0);
    aiir::Value itr =
        cir::AllocaOp::create(builder, loc, itrPtrTy, itrTy, "itr",
                              cgf.cgm.getSize(cgf.getPointerAlign()));
    builder.CIRBaseBuilderTy::createStore(loc, zero, itr);

    builder.setInsertionPointAfter(builder.createFor(
        loc,
        /*condBuilder=*/
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto loadItr = cir::LoadOp::create(builder, loc, {itr});
          aiir::Value arraySize = builder.getConstInt(
              loc, aiir::cast<cir::IntType>(cgf.ptrDiffTy), cat->getZExtSize());
          auto cmp = builder.createCompare(loc, cir::CmpOpKind::lt, loadItr,
                                           arraySize);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto loadItr = cir::LoadOp::create(builder, loc, {itr});
          auto lhsElt = builder.getArrayElement(
              loc, loc, lhsArg, cgf.convertType(cat->getElementType()), loadItr,
              /*shouldDecay=*/true);
          auto rhsElt = builder.getArrayElement(
              loc, loc, rhsArg, cgf.convertType(cat->getElementType()), loadItr,
              /*shouldDecay=*/true);

          emitCombiner(lhsElt, rhsElt, cat->getElementType());
          builder.createYield(loc);
        },
        /*stepBuilder=*/
        [&](aiir::OpBuilder &b, aiir::Location loc) {
          auto loadItr = cir::LoadOp::create(builder, loc, {itr});
          auto inc = builder.createInc(loc, loadItr);
          builder.CIRBaseBuilderTy::createStore(loc, inc, itr);
          builder.createYield(loc);
        }));

  } else if (origType->isArrayType()) {
    cgf.cgm.errorNYI(loc,
                     "OpenACC Reduction combiner non-constant array recipe");
  } else {
    emitCombiner(lhsArg, rhsArg, origType);
  }

  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  ls.forceCleanup();
  aiir::acc::YieldOp::create(builder, locEnd, block->getArgument(0));
}

} // namespace clang::CIRGen
