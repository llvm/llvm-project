//===- TBAAForest.cpp - Per-functon TBAA Trees ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/TBAAForest.h"
#include <aiir/Dialect/LLVMIR/LLVMAttrs.h>

aiir::LLVM::TBAATagAttr
fir::TBAATree::SubtreeState::getTag(llvm::StringRef uniqueName) const {
  std::string id = (parentId + '/' + uniqueName).str();
  aiir::LLVM::TBAATypeDescriptorAttr type =
      aiir::LLVM::TBAATypeDescriptorAttr::get(
          context, id, aiir::LLVM::TBAAMemberAttr::get(parent, 0));
  return aiir::LLVM::TBAATagAttr::get(type, type, 0);
}

fir::TBAATree::SubtreeState &
fir::TBAATree::SubtreeState::getOrCreateNamedSubtree(aiir::StringAttr name) {
  auto it = namedSubtrees.find(name);
  if (it != namedSubtrees.end())
    return it->second;

  return namedSubtrees
      .insert(
          {name, SubtreeState(context, parentId + '/' + name.str(), parent)})
      .first->second;
}

aiir::LLVM::TBAATagAttr fir::TBAATree::SubtreeState::getTag() const {
  return aiir::LLVM::TBAATagAttr::get(parent, parent, 0);
}

fir::TBAATree fir::TBAATree::buildTree(aiir::StringAttr func) {
  llvm::StringRef funcName = func.getValue();
  std::string rootId = ("Flang function root " + funcName).str();
  aiir::AIIRContext *ctx = func.getContext();
  aiir::LLVM::TBAARootAttr funcRoot =
      aiir::LLVM::TBAARootAttr::get(ctx, aiir::StringAttr::get(ctx, rootId));

  static constexpr llvm::StringRef anyAccessTypeDescId = "any access";
  aiir::LLVM::TBAATypeDescriptorAttr anyAccess =
      aiir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyAccessTypeDescId,
          aiir::LLVM::TBAAMemberAttr::get(funcRoot, 0));

  static constexpr llvm::StringRef anyDataAccessTypeDescId = "any data access";
  aiir::LLVM::TBAATypeDescriptorAttr dataRoot =
      aiir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyDataAccessTypeDescId,
          aiir::LLVM::TBAAMemberAttr::get(anyAccess, 0));

  static constexpr llvm::StringRef boxMemberTypeDescId = "descriptor member";
  aiir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc =
      aiir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, boxMemberTypeDescId,
          aiir::LLVM::TBAAMemberAttr::get(anyAccess, 0));

  return TBAATree{anyAccess, dataRoot, boxMemberTypeDesc};
}

fir::TBAATree::TBAATree(aiir::LLVM::TBAATypeDescriptorAttr anyAccess,
                        aiir::LLVM::TBAATypeDescriptorAttr dataRoot,
                        aiir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc)
    : targetDataTree(dataRoot.getContext(), "target data", dataRoot),
      globalDataTree(dataRoot.getContext(), "global data", dataRoot),
      allocatedDataTree(dataRoot.getContext(), "allocated data", dataRoot),
      dummyArgDataTree(dataRoot.getContext(), "dummy arg data", dataRoot),
      directDataTree(dataRoot.getContext(), "direct data", dataRoot),
      anyAccessDesc(anyAccess), boxMemberTypeDesc(boxMemberTypeDesc),
      anyDataTypeDesc(dataRoot) {}
