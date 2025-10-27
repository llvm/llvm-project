//===- TBAAForest.cpp - Per-functon TBAA Trees ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/TBAAForest.h"
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

mlir::LLVM::TBAATagAttr
fir::TBAATree::SubtreeState::getTag(llvm::StringRef uniqueName) const {
  std::string id = (parentId + '/' + uniqueName).str();
  mlir::LLVM::TBAATypeDescriptorAttr type =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          context, id, mlir::LLVM::TBAAMemberAttr::get(parent, 0));
  return mlir::LLVM::TBAATagAttr::get(type, type, 0);
}

fir::TBAATree::SubtreeState &
fir::TBAATree::SubtreeState::getOrCreateNamedSubtree(mlir::StringAttr name) {
  auto it = namedSubtrees.find(name);
  if (it != namedSubtrees.end())
    return it->second;

  return namedSubtrees
      .insert(
          {name, SubtreeState(context, parentId + '/' + name.str(), parent)})
      .first->second;
}

mlir::LLVM::TBAATagAttr fir::TBAATree::SubtreeState::getTag() const {
  return mlir::LLVM::TBAATagAttr::get(parent, parent, 0);
}

fir::TBAATree fir::TBAATree::buildTree(mlir::StringAttr func) {
  llvm::StringRef funcName = func.getValue();
  std::string rootId = ("Flang function root " + funcName).str();
  mlir::MLIRContext *ctx = func.getContext();
  mlir::LLVM::TBAARootAttr funcRoot =
      mlir::LLVM::TBAARootAttr::get(ctx, mlir::StringAttr::get(ctx, rootId));

  static constexpr llvm::StringRef anyAccessTypeDescId = "any access";
  mlir::LLVM::TBAATypeDescriptorAttr anyAccess =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyAccessTypeDescId,
          mlir::LLVM::TBAAMemberAttr::get(funcRoot, 0));

  static constexpr llvm::StringRef anyDataAccessTypeDescId = "any data access";
  mlir::LLVM::TBAATypeDescriptorAttr dataRoot =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyDataAccessTypeDescId,
          mlir::LLVM::TBAAMemberAttr::get(anyAccess, 0));

  static constexpr llvm::StringRef boxMemberTypeDescId = "descriptor member";
  mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, boxMemberTypeDescId,
          mlir::LLVM::TBAAMemberAttr::get(anyAccess, 0));

  return TBAATree{anyAccess, dataRoot, boxMemberTypeDesc};
}

fir::TBAATree::TBAATree(mlir::LLVM::TBAATypeDescriptorAttr anyAccess,
                        mlir::LLVM::TBAATypeDescriptorAttr dataRoot,
                        mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc)
    : targetDataTree(dataRoot.getContext(), "target data", dataRoot),
      globalDataTree(dataRoot.getContext(), "global data",
                     targetDataTree.getRoot()),
      allocatedDataTree(dataRoot.getContext(), "allocated data",
                        targetDataTree.getRoot()),
      dummyArgDataTree(dataRoot.getContext(), "dummy arg data", dataRoot),
      directDataTree(dataRoot.getContext(), "direct data",
                     targetDataTree.getRoot()),
      anyAccessDesc(anyAccess), boxMemberTypeDesc(boxMemberTypeDesc),
      anyDataTypeDesc(dataRoot) {}
