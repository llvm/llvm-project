//===-- UsedHelperDeclFinder.h - AST-based call graph for helper decls ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_USED_HELPER_DECL_FINDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_USED_HELPER_DECL_FINDER_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <vector>

namespace clang {
namespace move {

// A reference graph for finding used/unused helper declarations in a single
// translation unit (e.g. old.cc). We don't reuse CallGraph in clang/Analysis
// because that CallGraph only supports function declarations.
//
// Helper declarations include following types:
//   * function/variable/class definitions in an anonymous namespace.
//   * static function/variable definitions in a global/named namespace.
//
// The reference graph is a directed graph. Each node in the graph represents a
// helper declaration in old.cc or a non-moved/moved declaration (e.g. class,
// function) in old.h, which means each node is associated with a Decl.
//
// To construct the graph, we use AST matcher to find interesting Decls (usually
// a pair of Caller and Callee), and add an edge from the Caller node to the
// Callee node.
//
// Specially, for a class, it might have multiple declarations such methods
// and member variables. We only use a single node to present this class, and
// this node is associated with the class declaration (CXXRecordDecl).
//
// The graph has 3 types of edges:
//   1. moved_decl => helper_decl
//   2. non_moved_decl => helper_decl
//   3. helper_decl => helper_decl
class HelperDeclRefGraph {
public:
  HelperDeclRefGraph() = default;
  ~HelperDeclRefGraph() = default;

  // Add a directed edge from the caller node to the callee node.
  // A new node will be created if the node for Caller/Callee doesn't exist.
  //
  // Note that, all class member declarations are represented by a single node
  // in the graph. The corresponding Decl of this node is the class declaration.
  void addEdge(const Decl *Caller, const Decl *Callee);
  CallGraphNode *getNode(const Decl *D) const;

  // Get all reachable nodes in the graph from the given declaration D's node,
  // including D.
  llvm::DenseSet<const CallGraphNode *> getReachableNodes(const Decl *D) const;

  // Dump the call graph for debug purpose.
  void dump() const;

private:
  void print(raw_ostream &OS) const;
  // Lookup a node for the given declaration D. If not found, insert a new
  // node into the graph.
  CallGraphNode *getOrInsertNode(Decl *D);

  typedef llvm::DenseMap<const Decl *, std::unique_ptr<CallGraphNode>>
      DeclMapTy;

  // DeclMap owns all CallGraphNodes.
  DeclMapTy DeclMap;
};

// A builder helps to construct a call graph of helper declarations.
class HelperDeclRGBuilder : public ast_matchers::MatchFinder::MatchCallback {
public:
  HelperDeclRGBuilder() : RG(new HelperDeclRefGraph) {}
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  const HelperDeclRefGraph *getGraph() const { return RG.get(); }

  // Find out the outmost enclosing class/function declaration of a given D.
  // For a CXXMethodDecl, get its CXXRecordDecl; For a VarDecl/FunctionDecl, get
  // its outmost enclosing FunctionDecl or CXXRecordDecl.
  // Return D if not found.
  static const Decl *getOutmostClassOrFunDecl(const Decl *D);

private:
  std::unique_ptr<HelperDeclRefGraph> RG;
};

} // namespace move
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_USED_HELPER_DECL_FINDER_H
