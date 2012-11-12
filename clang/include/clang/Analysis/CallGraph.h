//== CallGraph.h - AST-based Call graph  ------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the AST-based CallGraph.
//
//  A call graph for functions whose definitions/bodies are available in the
//  current translation unit. The graph has a "virtual" root node that contains
//  edges to all externally available functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CALLGRAPH
#define LLVM_CLANG_ANALYSIS_CALLGRAPH

#include "clang/AST/DeclBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SetVector.h"

namespace clang {
class CallGraphNode;

/// \brief The AST-based call graph.
///
/// The call graph extends itself with the given declarations by implementing
/// the recursive AST visitor, which constructs the graph by visiting the given
/// declarations.
class CallGraph : public RecursiveASTVisitor<CallGraph> {
  friend class CallGraphNode;

  typedef llvm::DenseMap<const Decl *, CallGraphNode *> FunctionMapTy;

  /// FunctionMap owns all CallGraphNodes.
  FunctionMapTy FunctionMap;

  /// This is a virtual root node that has edges to all the global functions -
  /// 'main' or functions accessible from other translation units.
  CallGraphNode *Root;

  /// The list of nodes that have no parent. These are unreachable from Root.
  /// Declarations can get to this list due to impressions in the graph, for
  /// example, we do not track functions whose addresses were taken.
  llvm::SetVector<CallGraphNode *> ParentlessNodes;

public:
  CallGraph();
  ~CallGraph();

  /// \brief Populate the call graph with the functions in the given
  /// declaration.
  ///
  /// Recursively walks the declaration to find all the dependent Decls as well.
  void addToCallGraph(Decl *D) {
    TraverseDecl(D);
  }

  /// \brief Determine if a declaration should be included in the graph.
  static bool includeInGraph(const Decl *D);

  /// \brief Lookup the node for the given declaration.
  CallGraphNode *getNode(const Decl *) const;

  /// \brief Lookup the node for the given declaration. If none found, insert
  /// one into the graph.
  CallGraphNode *getOrInsertNode(Decl *);

  /// Iterators through all the elements in the graph. Note, this gives
  /// non-deterministic order.
  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;
  iterator begin() { return FunctionMap.begin(); }
  iterator end()   { return FunctionMap.end();   }
  const_iterator begin() const { return FunctionMap.begin(); }
  const_iterator end()   const { return FunctionMap.end();   }

  /// \brief Get the number of nodes in the graph.
  unsigned size() const { return FunctionMap.size(); }

  /// \ brief Get the virtual root of the graph, all the functions available
  /// externally are represented as callees of the node.
  CallGraphNode *getRoot() const { return Root; }

  /// Iterators through all the nodes of the graph that have no parent. These
  /// are the unreachable nodes, which are either unused or are due to us
  /// failing to add a call edge due to the analysis imprecision.
  typedef llvm::SetVector<CallGraphNode *>::iterator nodes_iterator;
  typedef llvm::SetVector<CallGraphNode *>::const_iterator const_nodes_iterator;
  nodes_iterator parentless_begin() { return ParentlessNodes.begin(); }
  nodes_iterator parentless_end() { return ParentlessNodes.end(); }
  const_nodes_iterator
    parentless_begin() const { return ParentlessNodes.begin(); }
  const_nodes_iterator
    parentless_end() const { return ParentlessNodes.end(); }

  void print(raw_ostream &os) const;
  void dump() const;
  void viewGraph() const;

  /// Part of recursive declaration visitation. We recursively visit all the
  /// Declarations to collect the root functions.
  bool VisitFunctionDecl(FunctionDecl *FD) {
    // We skip function template definitions, as their semantics is
    // only determined when they are instantiated.
    if (includeInGraph(FD))
      // If this function has external linkage, anything could call it.
      // Note, we are not precise here. For example, the function could have
      // its address taken.
      addNodeForDecl(FD, FD->isGlobal());
    return true;
  }

  /// Part of recursive declaration visitation.
  bool VisitObjCMethodDecl(ObjCMethodDecl *MD) {
    if (includeInGraph(MD))
      addNodeForDecl(MD, true);
    return true;
  }

  // We are only collecting the declarations, so do not step into the bodies.
  bool TraverseStmt(Stmt *S) { return true; }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

private:
  /// \brief Add the given declaration to the call graph.
  void addNodeForDecl(Decl *D, bool IsGlobal);

  /// \brief Allocate a new node in the graph.
  CallGraphNode *allocateNewNode(Decl *);
};

class CallGraphNode {
public:
  typedef CallGraphNode* CallRecord;

private:
  /// \brief The function/method declaration.
  Decl *FD;

  /// \brief The list of functions called from this node.
  // Small vector might be more efficient since we are only tracking functions
  // whose definition is in the current TU.
  llvm::SmallVector<CallRecord, 5> CalledFunctions;

public:
  CallGraphNode(Decl *D) : FD(D) {}

  typedef llvm::SmallVector<CallRecord, 5>::iterator iterator;
  typedef llvm::SmallVector<CallRecord, 5>::const_iterator const_iterator;

  /// Iterators through all the callees/children of the node.
  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end()   { return CalledFunctions.end(); }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end()   const { return CalledFunctions.end();   }

  inline bool empty() const {return CalledFunctions.empty(); }
  inline unsigned size() const {return CalledFunctions.size(); }

  void addCallee(CallGraphNode *N, CallGraph *CG) {
    CalledFunctions.push_back(N);
    CG->ParentlessNodes.remove(N);
  }

  Decl *getDecl() const { return FD; }

  StringRef getName() const;

  void print(raw_ostream &os) const;
  void dump() const;
};

} // end clang namespace

// Graph traits for iteration, viewing.
namespace llvm {
template <> struct GraphTraits<clang::CallGraphNode*> {
  typedef clang::CallGraphNode NodeType;
  typedef clang::CallGraphNode::CallRecord CallRecordTy;
  typedef std::pointer_to_unary_function<CallRecordTy,
                                         clang::CallGraphNode*> CGNDerefFun;
  static NodeType *getEntryNode(clang::CallGraphNode *CGN) { return CGN; }
  typedef mapped_iterator<NodeType::iterator, CGNDerefFun> ChildIteratorType;
  static inline ChildIteratorType child_begin(NodeType *N) {
    return map_iterator(N->begin(), CGNDerefFun(CGNDeref));
  }
  static inline ChildIteratorType child_end  (NodeType *N) {
    return map_iterator(N->end(), CGNDerefFun(CGNDeref));
  }
  static clang::CallGraphNode *CGNDeref(CallRecordTy P) {
    return P;
  }
};

template <> struct GraphTraits<const clang::CallGraphNode*> {
  typedef const clang::CallGraphNode NodeType;
  typedef NodeType::const_iterator ChildIteratorType;
  static NodeType *getEntryNode(const clang::CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};

template <> struct GraphTraits<clang::CallGraph*>
  : public GraphTraits<clang::CallGraphNode*> {

  static NodeType *getEntryNode(clang::CallGraph *CGN) {
    return CGN->getRoot();  // Start at the external node!
  }
  typedef std::pair<const clang::Decl*, clang::CallGraphNode*> PairTy;
  typedef std::pointer_to_unary_function<PairTy, clang::CallGraphNode&> DerefFun;
  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<clang::CallGraph::iterator, DerefFun> nodes_iterator;

  static nodes_iterator nodes_begin(clang::CallGraph *CG) {
    return map_iterator(CG->begin(), DerefFun(CGdereference));
  }
  static nodes_iterator nodes_end  (clang::CallGraph *CG) {
    return map_iterator(CG->end(), DerefFun(CGdereference));
  }
  static clang::CallGraphNode &CGdereference(PairTy P) {
    return *(P.second);
  }

  static unsigned size(clang::CallGraph *CG) {
    return CG->size();
  }
};

template <> struct GraphTraits<const clang::CallGraph*> :
  public GraphTraits<const clang::CallGraphNode*> {
  static NodeType *getEntryNode(const clang::CallGraph *CGN) {
    return CGN->getRoot();
  }
  typedef std::pair<const clang::Decl*, clang::CallGraphNode*> PairTy;
  typedef std::pointer_to_unary_function<PairTy, clang::CallGraphNode&> DerefFun;
  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<clang::CallGraph::const_iterator,
                          DerefFun> nodes_iterator;

  static nodes_iterator nodes_begin(const clang::CallGraph *CG) {
    return map_iterator(CG->begin(), DerefFun(CGdereference));
  }
  static nodes_iterator nodes_end(const clang::CallGraph *CG) {
    return map_iterator(CG->end(), DerefFun(CGdereference));
  }
  static clang::CallGraphNode &CGdereference(PairTy P) {
    return *(P.second);
  }

  static unsigned size(const clang::CallGraph *CG) {
    return CG->size();
  }
};

} // end llvm namespace

#endif
