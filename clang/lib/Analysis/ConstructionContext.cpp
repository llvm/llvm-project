//===- ConstructionContext.cpp - CFG constructor information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstructionContext class and its sub-classes,
// which represent various different ways of constructing C++ objects
// with the additional information the users may want to know about
// the constructor.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/ConstructionContext.h"

using namespace clang;

const ConstructionContextLayer *
ConstructionContextLayer::create(BumpVectorContext &C, TriggerTy Trigger,
                                 const ConstructionContextLayer *Parent) {
  ConstructionContextLayer *CC =
      C.getAllocator().Allocate<ConstructionContextLayer>();
  return new (CC) ConstructionContextLayer(Trigger, Parent);
}

bool ConstructionContextLayer::isStrictlyMoreSpecificThan(
    const ConstructionContextLayer *Other) const {
  const ConstructionContextLayer *Self = this;
  while (true) {
    if (!Other)
      return Self;
    if (!Self || !Self->isSameLayer(Other))
      return false;
    Self = Self->getParent();
    Other = Other->getParent();
  }
  llvm_unreachable("The above loop can only be terminated via return!");
}

const ConstructionContext *ConstructionContext::createFromLayers(
    BumpVectorContext &C, const ConstructionContextLayer *TopLayer) {
  // Before this point all we've had was a stockpile of arbitrary layers.
  // Now validate that it is shaped as one of the finite amount of expected
  // patterns.
  if (const Stmt *S = TopLayer->getTriggerStmt()) {
    if (const auto *DS = dyn_cast<DeclStmt>(S)) {
      assert(TopLayer->isLast());
      auto *CC =
          C.getAllocator().Allocate<SimpleVariableConstructionContext>();
      return new (CC) SimpleVariableConstructionContext(DS);
    } else if (const auto *NE = dyn_cast<CXXNewExpr>(S)) {
      assert(TopLayer->isLast());
      auto *CC =
          C.getAllocator().Allocate<NewAllocatedObjectConstructionContext>();
      return new (CC) NewAllocatedObjectConstructionContext(NE);
    } else if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(S)) {
      const MaterializeTemporaryExpr *MTE = nullptr;
      assert(BTE->getType().getCanonicalType()
                ->getAsCXXRecordDecl()->hasNonTrivialDestructor());
      // For temporaries with destructors, there may or may not be
      // lifetime extension on the parent layer.
      if (const ConstructionContextLayer *ParentLayer = TopLayer->getParent()) {
        assert(ParentLayer->isLast());
        MTE = cast<MaterializeTemporaryExpr>(ParentLayer->getTriggerStmt());
      }
      auto *CC =
          C.getAllocator().Allocate<TemporaryObjectConstructionContext>();
      return new (CC) TemporaryObjectConstructionContext(BTE, MTE);
    } else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(S)) {
      // If the object requires destruction and is not lifetime-extended,
      // then it must have a BTE within its MTE.
      assert(MTE->getType().getCanonicalType()
                ->getAsCXXRecordDecl()->hasTrivialDestructor() ||
             MTE->getStorageDuration() != SD_FullExpression);
      assert(TopLayer->isLast());
      auto *CC =
          C.getAllocator().Allocate<TemporaryObjectConstructionContext>();
      return new (CC) TemporaryObjectConstructionContext(nullptr, MTE);
    } else if (const auto *RS = dyn_cast<ReturnStmt>(S)) {
      assert(TopLayer->isLast());
      auto *CC =
          C.getAllocator().Allocate<ReturnedValueConstructionContext>();
      return new (CC) ReturnedValueConstructionContext(RS);
    }
  } else if (const CXXCtorInitializer *I = TopLayer->getTriggerInit()) {
    assert(TopLayer->isLast());
    auto *CC =
        C.getAllocator().Allocate<ConstructorInitializerConstructionContext>();
    return new (CC) ConstructorInitializerConstructionContext(I);
  }
  llvm_unreachable("Unexpected construction context!");
}
