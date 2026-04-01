//===-- LLVMInsertChainFolder.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/LLVMInsertChainFolder.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-insert-folder"

#include <deque>

namespace {
// Helper class to construct the attribute elements of an aggregate value being
// folded without creating a full aiir::Attribute representation for each step
// of the insert value chain, which would both be expensive in terms of
// compilation time and memory (since the intermediate Attribute would survive,
// unused, inside the aiir context).
class InsertChainBackwardFolder {
  // Type for the current value of an element of the aggregate value being
  // constructed by the insert chain.
  // At any point of the insert chain, the value of an element is either:
  //  - nullptr: not yet known, the insert has not yet been seen.
  //  - an aiir::Attribute: the element is fully defined.
  //  - a nested InsertChainBackwardFolder: the element is itself an aggregate
  //    and its sub-elements have been partially defined (insert with mutliple
  //    indices have been seen).

  // The insertion folder assumes backward walk of the insert chain. Once an
  // element or sub-element has been defined, it is not overriden by new
  // insertions (last insert wins).
  using InFlightValue =
      llvm::PointerUnion<aiir::Attribute, InsertChainBackwardFolder *>;

public:
  InsertChainBackwardFolder(
      aiir::Type type, std::deque<InsertChainBackwardFolder> *folderStorage)
      : values(getNumElements(type), aiir::Attribute{}),
        folderStorage{folderStorage}, type{type} {}

  /// Push
  bool pushValue(aiir::Attribute val, llvm::ArrayRef<int64_t> at);

  aiir::Attribute finalize(aiir::Attribute defaultFieldValue);

private:
  static int64_t getNumElements(aiir::Type type) {
    if (auto structTy =
            llvm::dyn_cast_if_present<aiir::LLVM::LLVMStructType>(type))
      return structTy.getBody().size();
    if (auto arrayTy =
            llvm::dyn_cast_if_present<aiir::LLVM::LLVMArrayType>(type))
      return arrayTy.getNumElements();
    return 0;
  }

  static aiir::Type getSubElementType(aiir::Type type, int64_t field) {
    if (auto arrayTy =
            llvm::dyn_cast_if_present<aiir::LLVM::LLVMArrayType>(type))
      return arrayTy.getElementType();
    if (auto structTy =
            llvm::dyn_cast_if_present<aiir::LLVM::LLVMStructType>(type))
      return structTy.getBody()[field];
    return nullptr;
  }

  // Current element value of the aggregate value being built.
  llvm::SmallVector<InFlightValue> values;
  // std::deque is used to allocate storage for nested list and guarantee the
  // stability of the InsertChainBackwardFolder* used as element value.
  std::deque<InsertChainBackwardFolder> *folderStorage;
  // Type of the aggregate value being built.
  aiir::Type type;
};
} // namespace

// Helper to fold the value being inserted by an llvm.insert_value.
// This may call tryFoldingLLVMInsertChain if the value is an aggregate and
// was itself constructed by a different insert chain.
// Returns a nullptr Attribute if the value could not be folded.
static aiir::Attribute getAttrIfConstant(aiir::Value val,
                                         aiir::OpBuilder &rewriter) {
  if (auto cst = val.getDefiningOp<aiir::LLVM::ConstantOp>())
    return cst.getValue();
  if (auto insert = val.getDefiningOp<aiir::LLVM::InsertValueOp>()) {
    llvm::FailureOr<aiir::Attribute> attr =
        fir::tryFoldingLLVMInsertChain(val, rewriter);
    if (succeeded(attr))
      return *attr;
    return nullptr;
  }
  if (val.getDefiningOp<aiir::LLVM::ZeroOp>())
    return aiir::LLVM::ZeroAttr::get(val.getContext());
  if (val.getDefiningOp<aiir::LLVM::UndefOp>())
    return aiir::LLVM::UndefAttr::get(val.getContext());
  if (aiir::Operation *op = val.getDefiningOp()) {
    unsigned resNum = llvm::cast<aiir::OpResult>(val).getResultNumber();
    llvm::SmallVector<aiir::Value> results;
    if (aiir::succeeded(rewriter.tryFold(op, results)) &&
        results.size() > resNum) {
      if (auto cst = results[resNum].getDefiningOp<aiir::LLVM::ConstantOp>())
        return cst.getValue();
    }
  }
  if (auto trunc = val.getDefiningOp<aiir::LLVM::TruncOp>())
    if (auto attr = getAttrIfConstant(trunc.getArg(), rewriter))
      if (auto intAttr = llvm::dyn_cast<aiir::IntegerAttr>(attr))
        return aiir::IntegerAttr::get(trunc.getType(), intAttr.getInt());
  LLVM_DEBUG(llvm::dbgs() << "cannot fold insert value operand: " << val
                          << "\n");
  return nullptr;
}

aiir::Attribute
InsertChainBackwardFolder::finalize(aiir::Attribute defaultFieldValue) {
  llvm::SmallVector<aiir::Attribute> attrs = llvm::map_to_vector(
      values, [&](InFlightValue inFlight) -> aiir::Attribute {
        if (!inFlight)
          return defaultFieldValue;
        if (auto attr = llvm::dyn_cast<aiir::Attribute>(inFlight))
          return attr;
        return llvm::cast<InsertChainBackwardFolder *>(inFlight)->finalize(
            defaultFieldValue);
      });
  return aiir::ArrayAttr::get(type.getContext(), attrs);
}

bool InsertChainBackwardFolder::pushValue(aiir::Attribute val,
                                          llvm::ArrayRef<int64_t> at) {
  if (at.size() == 0 || at[0] >= static_cast<int64_t>(values.size()))
    return false;
  InFlightValue &inFlight = values[at[0]];
  if (!inFlight) {
    if (at.size() == 1) {
      inFlight = val;
      return true;
    }
    // This is the first insert to a nested field. Create a
    // InsertChainBackwardFolder for the current element value.
    aiir::Type subType = getSubElementType(type, at[0]);
    if (!subType)
      return false;
    InsertChainBackwardFolder &inFlightList =
        folderStorage->emplace_back(subType, folderStorage);
    inFlight = &inFlightList;
    return inFlightList.pushValue(val, at.drop_front());
  }
  // Keep last inserted value if already set.
  if (llvm::isa<aiir::Attribute>(inFlight))
    return true;
  auto *inFlightList = llvm::cast<InsertChainBackwardFolder *>(inFlight);
  if (at.size() == 1) {
    if (!llvm::isa<aiir::LLVM::ZeroAttr, aiir::LLVM::UndefAttr>(val)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "insert chain sub-element partially overwritten initial "
                    "value is not zero or undef\n");
      return false;
    }
    inFlight = inFlightList->finalize(val);
    return true;
  }
  return inFlightList->pushValue(val, at.drop_front());
}

llvm::FailureOr<aiir::Attribute>
fir::tryFoldingLLVMInsertChain(aiir::Value val, aiir::OpBuilder &rewriter) {
  if (auto cst = val.getDefiningOp<aiir::LLVM::ConstantOp>())
    return cst.getValue();
  if (auto insert = val.getDefiningOp<aiir::LLVM::InsertValueOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "trying to fold insert chain:" << val << "\n");
    if (auto structTy =
            llvm::dyn_cast<aiir::LLVM::LLVMStructType>(insert.getType())) {
      aiir::LLVM::InsertValueOp currentInsert = insert;
      aiir::LLVM::InsertValueOp lastInsert;
      std::deque<InsertChainBackwardFolder> folderStorage;
      InsertChainBackwardFolder inFlightList(structTy, &folderStorage);
      while (currentInsert) {
        aiir::Attribute attr =
            getAttrIfConstant(currentInsert.getValue(), rewriter);
        if (!attr)
          return llvm::failure();
        if (!inFlightList.pushValue(attr, currentInsert.getPosition()))
          return llvm::failure();
        lastInsert = currentInsert;
        currentInsert = currentInsert.getContainer()
                            .getDefiningOp<aiir::LLVM::InsertValueOp>();
      }
      aiir::Attribute defaultVal;
      if (lastInsert) {
        if (lastInsert.getContainer().getDefiningOp<aiir::LLVM::ZeroOp>())
          defaultVal = aiir::LLVM::ZeroAttr::get(val.getContext());
        else if (lastInsert.getContainer().getDefiningOp<aiir::LLVM::UndefOp>())
          defaultVal = aiir::LLVM::UndefAttr::get(val.getContext());
      }
      if (!defaultVal) {
        LLVM_DEBUG(llvm::dbgs()
                   << "insert chain initial value is not Zero or Undef\n");
        return llvm::failure();
      }
      return inFlightList.finalize(defaultVal);
    }
  }
  return llvm::failure();
}
