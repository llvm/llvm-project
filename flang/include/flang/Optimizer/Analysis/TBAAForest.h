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

    /// Create a TBAA tag pointing to the root of this subtree,
    /// i.e. all the children tags will alias with this tag.
    mlir::LLVM::TBAATagAttr getTag() const;

    mlir::LLVM::TBAATypeDescriptorAttr getRoot() const { return parent; }

    /// For the given name, get or create a subtree in the current
    /// subtree. For example, this is used for creating subtrees
    /// inside the "global data" subtree for the COMMON block variables
    /// belonging to the same COMMON block.
    SubtreeState &getOrCreateNamedSubtree(mlir::StringAttr name);

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
    // A map of named sub-trees, e.g. sub-trees of the COMMON blocks
    // placed under the "global data" root.
    llvm::DenseMap<mlir::StringAttr, SubtreeState> namedSubtrees;
  };

  /// A subtree for POINTER/TARGET variables data.
  /// Any POINTER variable must use a tag that points
  /// to the root of this subtree.
  /// A TARGET dummy argument must also point to this root.
  SubtreeState targetDataTree;
  /// A subtree for global variables data (e.g. user module variables).
  SubtreeState globalDataTree;
  /// A subtree for variables allocated via fir.alloca or fir.allocmem.
  SubtreeState allocatedDataTree;
  /// A subtree for subprogram's dummy arguments.
  /// It only contains children for the dummy arguments
  /// that are not POINTER/TARGET. They all do not conflict
  /// with each other and with any other data access, except
  /// with unknown data accesses (FIR alias analysis uses
  /// SourceKind::Indirect for sources of such accesses).
  SubtreeState dummyArgDataTree;
  /// A subtree for global variables descriptors.
  SubtreeState directDataTree;
  mlir::LLVM::TBAATypeDescriptorAttr anyAccessDesc;
  mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc;
  mlir::LLVM::TBAATypeDescriptorAttr anyDataTypeDesc;

  // Structure of the created tree:
  //   Function root
  //   |
  //   "any access"
  //   |
  //   |- "descriptor member"
  //   |- "any data access"
  //      |
  //      |- "dummy arg data"
  //      |- "target data"
  //         |
  //         |- "allocated data"
  //         |- "direct data"
  //         |- "global data"
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
      return getFuncTree(mlir::cast<mlir::StringAttr>(attr));
    }
    return getFuncTree(func.getSymNameAttr());
  }
  // Returns the TBAA tree associated with the scope enclosed
  // within the given function. With MLIR inlining, there may
  // be multiple scopes within a single function. It is the caller's
  // responsibility to provide unique name for the scope.
  // If the scope string is empty, returns the TBAA tree for the
  // "root" scope of the given function.
  inline TBAATree &getMutableFuncTreeWithScope(mlir::func::FuncOp func,
                                               llvm::StringRef scope) {
    mlir::StringAttr name = func.getSymNameAttr();
    if (!scope.empty())
      name = mlir::StringAttr::get(name.getContext(),
                                   llvm::Twine(name) + " - " + scope);
    return getFuncTree(name);
  }

  inline const TBAATree &getFuncTreeWithScope(mlir::func::FuncOp func,
                                              llvm::StringRef scope) {
    return getMutableFuncTreeWithScope(func, scope);
  }

private:
  TBAATree &getFuncTree(mlir::StringAttr symName) {
    if (!separatePerFunction)
      symName = mlir::StringAttr::get(symName.getContext(), "");
    if (!trees.contains(symName))
      trees.insert({symName, TBAATree::buildTree(symName)});
    auto it = trees.find(symName);
    assert(it != trees.end());
    return it->second;
  }

  // Should each function use a different tree?
  const bool separatePerFunction;
  // TBAA tree per function
  llvm::DenseMap<mlir::StringAttr, TBAATree> trees;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H
