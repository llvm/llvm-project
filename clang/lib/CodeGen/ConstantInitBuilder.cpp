//===--- ConstantInitBuilder.cpp - Global initializer builder -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines out-of-line routines for building initializers for
// global variables, in particular the kind of globals that are implicitly
// introduced by various language ABIs.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ConstantInitBuilder.h"
#include "CodeGenModule.h"

using namespace clang;
using namespace CodeGen;

llvm::GlobalVariable *
ConstantInitBuilderBase::createGlobal(llvm::Constant *initializer,
                                      const llvm::Twine &name,
                                      CharUnits alignment,
                                      bool constant,
                                      llvm::GlobalValue::LinkageTypes linkage,
                                      unsigned addressSpace) {
  auto GV = new llvm::GlobalVariable(CGM.getModule(),
                                     initializer->getType(),
                                     constant,
                                     linkage,
                                     initializer,
                                     name,
                                     /*insert before*/ nullptr,
                                     llvm::GlobalValue::NotThreadLocal,
                                     addressSpace);
  GV->setAlignment(alignment.getQuantity());
  resolveSelfReferences(GV);
  return GV;
}

void ConstantInitBuilderBase::setGlobalInitializer(llvm::GlobalVariable *GV,
                                                   llvm::Constant *initializer){
  GV->setInitializer(initializer);

  if (!SelfReferences.empty())
    resolveSelfReferences(GV);
}

void ConstantInitBuilderBase::resolveSelfReferences(llvm::GlobalVariable *GV) {
  for (auto &entry : SelfReferences) {
    llvm::Constant *resolvedReference =
      llvm::ConstantExpr::getInBoundsGetElementPtr(
        GV->getValueType(), GV, entry.Indices);
    entry.Dummy->replaceAllUsesWith(resolvedReference);
    entry.Dummy->eraseFromParent();
  }
}

void ConstantAggregateBuilderBase::addSize(CharUnits size) {
  add(Builder.CGM.getSize(size));
}

llvm::Constant *
ConstantAggregateBuilderBase::getRelativeOffset(llvm::IntegerType *offsetType,
                                                llvm::Constant *target) {
  // Compute the address of the relative-address slot.
  auto base = getAddrOfCurrentPosition(offsetType);

  // Subtract.
  base = llvm::ConstantExpr::getPtrToInt(base, Builder.CGM.IntPtrTy);
  target = llvm::ConstantExpr::getPtrToInt(target, Builder.CGM.IntPtrTy);
  llvm::Constant *offset = llvm::ConstantExpr::getSub(target, base);

  // Truncate to the relative-address type if necessary.
  if (Builder.CGM.IntPtrTy != offsetType) {
    offset = llvm::ConstantExpr::getTrunc(offset, offsetType);
  }

  return offset;
}

llvm::Constant *
ConstantAggregateBuilderBase::getAddrOfCurrentPosition(llvm::Type *type) {
  // Make a global variable.  We will replace this with a GEP to this
  // position after installing the initializer.
  auto dummy =
    new llvm::GlobalVariable(Builder.CGM.getModule(), type, true,
                             llvm::GlobalVariable::PrivateLinkage,
                             nullptr, "");
  Builder.SelfReferences.emplace_back(dummy);
  auto &entry = Builder.SelfReferences.back();
  (void) getGEPIndicesToCurrentPosition(entry.Indices);
  return dummy;
}

void ConstantAggregateBuilderBase::getGEPIndicesTo(
                               llvm::SmallVectorImpl<llvm::Constant*> &indices,
                               size_t position) const {
  // Recurse on the parent builder if present.
  if (Parent) {
    Parent->getGEPIndicesTo(indices, Begin);

  // Otherwise, add an index to drill into the first level of pointer. 
  } else {
    assert(indices.empty());
    indices.push_back(llvm::ConstantInt::get(Builder.CGM.Int32Ty, 0));
  }

  assert(position >= Begin);
  // We have to use i32 here because struct GEPs demand i32 indices.
  // It's rather unlikely to matter in practice.
  indices.push_back(llvm::ConstantInt::get(Builder.CGM.Int32Ty,
                                           position - Begin));
}

CharUnits ConstantAggregateBuilderBase::getOffsetFromGlobalTo(size_t end) const{
  size_t cacheEnd = CachedOffsetEnd;
  assert(cacheEnd <= end);

  // Fast path: if the cache is valid, just use it.
  if (cacheEnd == end) {
    return CachedOffsetFromGlobal;
  }

  // If the cached range ends before the index at which the current
  // aggregate starts, recurse for the parent.
  CharUnits offset;
  if (cacheEnd < Begin) {
    assert(cacheEnd == 0);
    assert(Parent && "Begin != 0 for root builder");
    cacheEnd = Begin;
    offset = Parent->getOffsetFromGlobalTo(Begin);
  } else {
    offset = CachedOffsetFromGlobal;
  }

  // Perform simple layout on the elements in cacheEnd..<end.
  if (cacheEnd != end) {
    auto &layout = Builder.CGM.getDataLayout();
    do {
      llvm::Constant *element = Builder.Buffer[cacheEnd];
      assert(element != nullptr &&
             "cannot compute offset when a placeholder is present");
      llvm::Type *elementType = element->getType();
      offset = offset.alignTo(CharUnits::fromQuantity(
                                layout.getABITypeAlignment(elementType)));
      offset += CharUnits::fromQuantity(layout.getTypeStoreSize(elementType));
    } while (++cacheEnd != end);
  }

  // Cache and return.
  CachedOffsetEnd = cacheEnd;
  CachedOffsetFromGlobal = offset;
  return offset;
}

llvm::Constant *ConstantAggregateBuilderBase::finishArray(llvm::Type *eltTy) {
  markFinished();

  auto &buffer = getBuffer();
  assert((Begin < buffer.size() ||
          (Begin == buffer.size() && eltTy))
         && "didn't add any array elements without element type");
  auto elts = llvm::makeArrayRef(buffer).slice(Begin);
  if (!eltTy) eltTy = elts[0]->getType();
  auto type = llvm::ArrayType::get(eltTy, elts.size());
  auto constant = llvm::ConstantArray::get(type, elts);
  buffer.erase(buffer.begin() + Begin, buffer.end());
  return constant;
}

llvm::Constant *
ConstantAggregateBuilderBase::finishStruct(llvm::StructType *ty) {
  markFinished();

  auto &buffer = getBuffer();
  assert(Begin < buffer.size() && "didn't add any struct elements?");
  auto elts = llvm::makeArrayRef(buffer).slice(Begin);

  llvm::Constant *constant;
  if (ty) {
    constant = llvm::ConstantStruct::get(ty, elts);
  } else {
    constant = llvm::ConstantStruct::getAnon(elts, /*packed*/ false);
  }

  buffer.erase(buffer.begin() + Begin, buffer.end());
  return constant;
}
