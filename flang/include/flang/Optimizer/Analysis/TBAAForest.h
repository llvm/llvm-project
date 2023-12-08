//===-- TBAAForest.h - A TBAA tree for each function -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H
#define FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H

#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace fir {

//===----------------------------------------------------------------------===//
// TBAATree
//===----------------------------------------------------------------------===//
/// Per-function TBAA tree. Each tree contains branches for data (of various
/// kinds) and descriptor access
struct TBAATree {
  //===----------------------------------------------------------------------===//
  // TBAAForrest::TBAATree::SubtreeState
  //===----------------------------------------------------------------------===//
  /// This contains a TBAA subtree based on some parent. New tags can be added
  /// under the parent using getTag.
  class SubtreeState {
    friend TBAATree; // only allow construction by TBAATree
  public:
    SubtreeState() = delete;
    SubtreeState(const SubtreeState &) = delete;
    SubtreeState(SubtreeState &&) = default;

    mlir::LLVM::TBAATagAttr getTag(llvm::StringRef uniqueId) const;

  private:
    SubtreeState(mlir::MLIRContext *ctx, std::string name,
                 mlir::LLVM::TBAANodeAttr grandParent)
        : parentId{std::move(name)}, context(ctx) {
      parent = mlir::LLVM::TBAATypeDescriptorAttr::get(
          context, parentId, mlir::LLVM::TBAAMemberAttr::get(grandParent, 0));
    }

    const std::string parentId;
    mlir::MLIRContext *const context;
    mlir::LLVM::TBAATypeDescriptorAttr parent;
    llvm::DenseMap<llvm::StringRef, mlir::LLVM::TBAATagAttr> tagDedup;
  };

  SubtreeState globalDataTree;
  SubtreeState allocatedDataTree;
  SubtreeState dummyArgDataTree;
  SubtreeState directDataTree;
  mlir::LLVM::TBAATypeDescriptorAttr anyAccessDesc;
  mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc;
  mlir::LLVM::TBAATypeDescriptorAttr anyDataTypeDesc;

  static TBAATree buildTree(mlir::StringAttr functionName);

private:
  TBAATree(mlir::LLVM::TBAATypeDescriptorAttr anyAccess,
           mlir::LLVM::TBAATypeDescriptorAttr dataRoot,
           mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc);
};

//===----------------------------------------------------------------------===//
// TBAAForrest
//===----------------------------------------------------------------------===//
/// Collection of TBAATrees, usually indexed by function (so that each function
/// has a different TBAATree)
class TBAAForrest {
public:
  explicit TBAAForrest(bool separatePerFunction = true)
      : separatePerFunction{separatePerFunction} {}

  inline const TBAATree &operator[](mlir::func::FuncOp func) {
    return getFuncTree(func.getSymNameAttr());
  }
  inline const TBAATree &operator[](mlir::LLVM::LLVMFuncOp func) {
    // the external name conversion pass may rename some functions. Their old
    // name must be used so that we add to the tbaa tree added in the FIR pass
    mlir::Attribute attr = func->getAttr(getInternalFuncNameAttrName());
    if (attr) {
      return getFuncTree(attr.cast<mlir::StringAttr>());
    }
    return getFuncTree(func.getSymNameAttr());
  }

private:
  const TBAATree &getFuncTree(mlir::StringAttr symName) {
    if (!separatePerFunction)
      symName = mlir::StringAttr::get(symName.getContext(), "");
    if (!trees.contains(symName))
      trees.insert({symName, TBAATree::buildTree(symName)});
    return trees.at(symName);
  }

  // Should each function use a different tree?
  const bool separatePerFunction;
  // TBAA tree per function
  llvm::DenseMap<mlir::StringAttr, TBAATree> trees;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H
