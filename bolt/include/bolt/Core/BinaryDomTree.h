//==- bolt/Core/BinaryDomTree.h - Dominator Tree at low-level IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the BinaryDomTree class, which represents a dominator tree
// in the CFG of a binary function.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BINARY_DOMTREE_H
#define BOLT_CORE_BINARY_DOMTREE_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

namespace llvm {
namespace bolt {

using BinaryDomTreeNode = DomTreeNodeBase<BinaryBasicBlock>;
using BinaryDominatorTree = DomTreeBase<BinaryBasicBlock>;

} // namespace bolt

// BinaryDominatorTree GraphTraits specializations.
template <>
struct GraphTraits<bolt::BinaryDomTreeNode *>
    : public DomTreeGraphTraitsBase<bolt::BinaryDomTreeNode,
                                    bolt::BinaryDomTreeNode::iterator> {};

template <>
struct GraphTraits<const bolt::BinaryDomTreeNode *>
    : public DomTreeGraphTraitsBase<const bolt::BinaryDomTreeNode,
                                    bolt::BinaryDomTreeNode::const_iterator> {};

template <>
struct GraphTraits<bolt::BinaryDominatorTree *>
    : public GraphTraits<bolt::BinaryDomTreeNode *> {
  static NodeRef getEntryNode(bolt::BinaryDominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(bolt::BinaryDominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(bolt::BinaryDominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

} // namespace llvm

#endif
