//===- RegionGraphTraits.h - llvm::GraphTraits for CFGs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements specializations of llvm::GraphTraits for various AIIR
// CFG data types.  This allows the generic LLVM graph algorithms to be applied
// to CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_REGIONGRAPHTRAITS_H
#define AIIR_IR_REGIONGRAPHTRAITS_H

#include "aiir/IR/Region.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {
template <>
struct GraphTraits<aiir::Block *> {
  using ChildIteratorType = aiir::Block::succ_iterator;
  using Node = aiir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }
};

template <>
struct GraphTraits<Inverse<aiir::Block *>> {
  using ChildIteratorType = aiir::Block::pred_iterator;
  using Node = aiir::Block;
  using NodeRef = Node *;
  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->pred_end();
  }
};

template <>
struct GraphTraits<const aiir::Block *> {
  using ChildIteratorType = aiir::Block::succ_iterator;
  using Node = const aiir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<aiir::Block *>(node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<aiir::Block *>(node)->succ_end();
  }
};

template <>
struct GraphTraits<Inverse<const aiir::Block *>> {
  using ChildIteratorType = aiir::Block::pred_iterator;
  using Node = const aiir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }

  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<aiir::Block *>(node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<aiir::Block *>(node)->pred_end();
  }
};

template <>
struct GraphTraits<aiir::Region *> : public GraphTraits<aiir::Block *> {
  using GraphType = aiir::Region *;
  using NodeRef = aiir::Block *;

  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }

  using nodes_iterator = pointer_iterator<aiir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};

template <>
struct GraphTraits<Inverse<aiir::Region *>>
    : public GraphTraits<Inverse<aiir::Block *>> {
  using GraphType = Inverse<aiir::Region *>;
  using NodeRef = NodeRef;

  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }

  using nodes_iterator = pointer_iterator<aiir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};

} // namespace llvm

#endif
