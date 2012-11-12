//===--- StmtGraphTraits.h - Graph Traits for the class Stmt ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a template specialization of llvm::GraphTraits to
//  treat ASTs (Stmt*) as graphs
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_GRAPHTRAITS_H
#define LLVM_CLANG_AST_STMT_GRAPHTRAITS_H

#include "clang/AST/Stmt.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/DepthFirstIterator.h"

namespace llvm {

//template <typename T> struct GraphTraits;


template <> struct GraphTraits<clang::Stmt*> {
  typedef clang::Stmt                       NodeType;
  typedef clang::Stmt::child_iterator       ChildIteratorType;
  typedef llvm::df_iterator<clang::Stmt*>   nodes_iterator;

  static NodeType* getEntryNode(clang::Stmt* S) { return S; }

  static inline ChildIteratorType child_begin(NodeType* N) {
    if (N) return N->child_begin();
    else return ChildIteratorType();
  }

  static inline ChildIteratorType child_end(NodeType* N) {
    if (N) return N->child_end();
    else return ChildIteratorType();
  }

  static nodes_iterator nodes_begin(clang::Stmt* S) {
    return df_begin(S);
  }

  static nodes_iterator nodes_end(clang::Stmt* S) {
    return df_end(S);
  }
};


template <> struct GraphTraits<const clang::Stmt*> {
  typedef const clang::Stmt                       NodeType;
  typedef clang::Stmt::const_child_iterator       ChildIteratorType;
  typedef llvm::df_iterator<const clang::Stmt*>   nodes_iterator;

  static NodeType* getEntryNode(const clang::Stmt* S) { return S; }

  static inline ChildIteratorType child_begin(NodeType* N) {
    if (N) return N->child_begin();
    else return ChildIteratorType();
  }

  static inline ChildIteratorType child_end(NodeType* N) {
    if (N) return N->child_end();
    else return ChildIteratorType();
  }

  static nodes_iterator nodes_begin(const clang::Stmt* S) {
    return df_begin(S);
  }

  static nodes_iterator nodes_end(const clang::Stmt* S) {
    return df_end(S);
  }
};


} // end namespace llvm

#endif
