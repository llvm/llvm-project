//=-- ExplodedGraph.cpp - Local, Path-Sens. "Exploded Graph" -*- C++ -*------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template classes ExplodedNode and ExplodedGraph,
//  which represent a path-sensitive, intra-procedural "exploded graph."
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ParentMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <vector>

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Node auditing.
//===----------------------------------------------------------------------===//

// An out of line virtual method to provide a home for the class vtable.
ExplodedNode::Auditor::~Auditor() {}

#ifndef NDEBUG
static ExplodedNode::Auditor* NodeAuditor = 0;
#endif

void ExplodedNode::SetAuditor(ExplodedNode::Auditor* A) {
#ifndef NDEBUG
  NodeAuditor = A;
#endif
}

//===----------------------------------------------------------------------===//
// Cleanup.
//===----------------------------------------------------------------------===//

ExplodedGraph::ExplodedGraph()
  : NumNodes(0), ReclaimNodeInterval(0) {}

ExplodedGraph::~ExplodedGraph() {}

//===----------------------------------------------------------------------===//
// Node reclamation.
//===----------------------------------------------------------------------===//

bool ExplodedGraph::shouldCollect(const ExplodedNode *node) {
  // Reclaim all nodes that match *all* the following criteria:
  //
  // (1) 1 predecessor (that has one successor)
  // (2) 1 successor (that has one predecessor)
  // (3) The ProgramPoint is for a PostStmt, but not a PostStore.
  // (4) There is no 'tag' for the ProgramPoint.
  // (5) The 'store' is the same as the predecessor.
  // (6) The 'GDM' is the same as the predecessor.
  // (7) The LocationContext is the same as the predecessor.
  // (8) The PostStmt isn't for a non-consumed Stmt or Expr.
  // (9) The successor is not a CallExpr StmtPoint (so that we would be able to
  //     find it when retrying a call with no inlining).
  // FIXME: It may be safe to reclaim PreCall and PostCall nodes as well.

  // Conditions 1 and 2.
  if (node->pred_size() != 1 || node->succ_size() != 1)
    return false;

  const ExplodedNode *pred = *(node->pred_begin());
  if (pred->succ_size() != 1)
    return false;
  
  const ExplodedNode *succ = *(node->succ_begin());
  if (succ->pred_size() != 1)
    return false;

  // Condition 3.
  ProgramPoint progPoint = node->getLocation();
  if (!isa<PostStmt>(progPoint) || isa<PostStore>(progPoint))
    return false;

  // Condition 4.
  PostStmt ps = cast<PostStmt>(progPoint);
  if (ps.getTag())
    return false;

  // Conditions 5, 6, and 7.
  ProgramStateRef state = node->getState();
  ProgramStateRef pred_state = pred->getState();    
  if (state->store != pred_state->store || state->GDM != pred_state->GDM ||
      progPoint.getLocationContext() != pred->getLocationContext())
    return false;
  
  // Condition 8.
  // Do not collect nodes for non-consumed Stmt or Expr to ensure precise
  // diagnostic generation; specifically, so that we could anchor arrows
  // pointing to the beginning of statements (as written in code).
  if (!isa<Expr>(ps.getStmt()))
    return false;
  
  if (const Expr *Ex = dyn_cast<Expr>(ps.getStmt())) {
    ParentMap &PM = progPoint.getLocationContext()->getParentMap();
    if (!PM.isConsumedExpr(Ex))
      return false;
  }
  
  // Condition 9.
  const ProgramPoint SuccLoc = succ->getLocation();
  if (const StmtPoint *SP = dyn_cast<StmtPoint>(&SuccLoc))
    if (CallEvent::isCallStmt(SP->getStmt()))
      return false;

  return true;
}

void ExplodedGraph::collectNode(ExplodedNode *node) {
  // Removing a node means:
  // (a) changing the predecessors successor to the successor of this node
  // (b) changing the successors predecessor to the predecessor of this node
  // (c) Putting 'node' onto freeNodes.
  assert(node->pred_size() == 1 || node->succ_size() == 1);
  ExplodedNode *pred = *(node->pred_begin());
  ExplodedNode *succ = *(node->succ_begin());
  pred->replaceSuccessor(succ);
  succ->replacePredecessor(pred);
  FreeNodes.push_back(node);
  Nodes.RemoveNode(node);
  --NumNodes;
  node->~ExplodedNode();  
}

void ExplodedGraph::reclaimRecentlyAllocatedNodes() {
  if (ChangedNodes.empty())
    return;

  // Only periodically reclaim nodes so that we can build up a set of
  // nodes that meet the reclamation criteria.  Freshly created nodes
  // by definition have no successor, and thus cannot be reclaimed (see below).
  assert(ReclaimCounter > 0);
  if (--ReclaimCounter != 0)
    return;
  ReclaimCounter = ReclaimNodeInterval;

  for (NodeVector::iterator it = ChangedNodes.begin(), et = ChangedNodes.end();
       it != et; ++it) {
    ExplodedNode *node = *it;
    if (shouldCollect(node))
      collectNode(node);
  }
  ChangedNodes.clear();
}

//===----------------------------------------------------------------------===//
// ExplodedNode.
//===----------------------------------------------------------------------===//

// An NodeGroup's storage type is actually very much like a TinyPtrVector:
// it can be either a pointer to a single ExplodedNode, or a pointer to a
// BumpVector allocated with the ExplodedGraph's allocator. This allows the
// common case of single-node NodeGroups to be implemented with no extra memory.
//
// Consequently, each of the NodeGroup methods have up to four cases to handle:
// 1. The flag is set and this group does not actually contain any nodes.
// 2. The group is empty, in which case the storage value is null.
// 3. The group contains a single node.
// 4. The group contains more than one node.
typedef BumpVector<ExplodedNode *> ExplodedNodeVector;
typedef llvm::PointerUnion<ExplodedNode *, ExplodedNodeVector *> GroupStorage;

void ExplodedNode::addPredecessor(ExplodedNode *V, ExplodedGraph &G) {
  assert (!V->isSink());
  Preds.addNode(V, G);
  V->Succs.addNode(this, G);
#ifndef NDEBUG
  if (NodeAuditor) NodeAuditor->AddEdge(V, this);
#endif
}

void ExplodedNode::NodeGroup::replaceNode(ExplodedNode *node) {
  assert(!getFlag());

  GroupStorage &Storage = reinterpret_cast<GroupStorage&>(P);
  assert(Storage.is<ExplodedNode *>());
  Storage = node;
  assert(Storage.is<ExplodedNode *>());
}

void ExplodedNode::NodeGroup::addNode(ExplodedNode *N, ExplodedGraph &G) {
  assert(!getFlag());

  GroupStorage &Storage = reinterpret_cast<GroupStorage&>(P);
  if (Storage.isNull()) {
    Storage = N;
    assert(Storage.is<ExplodedNode *>());
    return;
  }

  ExplodedNodeVector *V = Storage.dyn_cast<ExplodedNodeVector *>();

  if (!V) {
    // Switch from single-node to multi-node representation.
    ExplodedNode *Old = Storage.get<ExplodedNode *>();

    BumpVectorContext &Ctx = G.getNodeAllocator();
    V = G.getAllocator().Allocate<ExplodedNodeVector>();
    new (V) ExplodedNodeVector(Ctx, 4);
    V->push_back(Old, Ctx);

    Storage = V;
    assert(!getFlag());
    assert(Storage.is<ExplodedNodeVector *>());
  }

  V->push_back(N, G.getNodeAllocator());
}

unsigned ExplodedNode::NodeGroup::size() const {
  if (getFlag())
    return 0;

  const GroupStorage &Storage = reinterpret_cast<const GroupStorage &>(P);
  if (Storage.isNull())
    return 0;
  if (ExplodedNodeVector *V = Storage.dyn_cast<ExplodedNodeVector *>())
    return V->size();
  return 1;
}

ExplodedNode * const *ExplodedNode::NodeGroup::begin() const {
  if (getFlag())
    return 0;

  const GroupStorage &Storage = reinterpret_cast<const GroupStorage &>(P);
  if (Storage.isNull())
    return 0;
  if (ExplodedNodeVector *V = Storage.dyn_cast<ExplodedNodeVector *>())
    return V->begin();
  return Storage.getAddrOfPtr1();
}

ExplodedNode * const *ExplodedNode::NodeGroup::end() const {
  if (getFlag())
    return 0;

  const GroupStorage &Storage = reinterpret_cast<const GroupStorage &>(P);
  if (Storage.isNull())
    return 0;
  if (ExplodedNodeVector *V = Storage.dyn_cast<ExplodedNodeVector *>())
    return V->end();
  return Storage.getAddrOfPtr1() + 1;
}

ExplodedNode *ExplodedGraph::getNode(const ProgramPoint &L,
                                     ProgramStateRef State,
                                     bool IsSink,
                                     bool* IsNew) {
  // Profile 'State' to determine if we already have an existing node.
  llvm::FoldingSetNodeID profile;
  void *InsertPos = 0;

  NodeTy::Profile(profile, L, State, IsSink);
  NodeTy* V = Nodes.FindNodeOrInsertPos(profile, InsertPos);

  if (!V) {
    if (!FreeNodes.empty()) {
      V = FreeNodes.back();
      FreeNodes.pop_back();
    }
    else {
      // Allocate a new node.
      V = (NodeTy*) getAllocator().Allocate<NodeTy>();
    }

    new (V) NodeTy(L, State, IsSink);

    if (ReclaimNodeInterval)
      ChangedNodes.push_back(V);

    // Insert the node into the node set and return it.
    Nodes.InsertNode(V, InsertPos);
    ++NumNodes;

    if (IsNew) *IsNew = true;
  }
  else
    if (IsNew) *IsNew = false;

  return V;
}

std::pair<ExplodedGraph*, InterExplodedGraphMap*>
ExplodedGraph::Trim(const NodeTy* const* NBeg, const NodeTy* const* NEnd,
               llvm::DenseMap<const void*, const void*> *InverseMap) const {

  if (NBeg == NEnd)
    return std::make_pair((ExplodedGraph*) 0,
                          (InterExplodedGraphMap*) 0);

  assert (NBeg < NEnd);

  OwningPtr<InterExplodedGraphMap> M(new InterExplodedGraphMap());

  ExplodedGraph* G = TrimInternal(NBeg, NEnd, M.get(), InverseMap);

  return std::make_pair(static_cast<ExplodedGraph*>(G), M.take());
}

ExplodedGraph*
ExplodedGraph::TrimInternal(const ExplodedNode* const* BeginSources,
                            const ExplodedNode* const* EndSources,
                            InterExplodedGraphMap* M,
                   llvm::DenseMap<const void*, const void*> *InverseMap) const {

  typedef llvm::DenseSet<const ExplodedNode*> Pass1Ty;
  Pass1Ty Pass1;

  typedef llvm::DenseMap<const ExplodedNode*, ExplodedNode*> Pass2Ty;
  Pass2Ty& Pass2 = M->M;

  SmallVector<const ExplodedNode*, 10> WL1, WL2;

  // ===- Pass 1 (reverse DFS) -===
  for (const ExplodedNode* const* I = BeginSources; I != EndSources; ++I) {
    if (*I)
      WL1.push_back(*I);
  }

  // Process the first worklist until it is empty.  Because it is a std::list
  // it acts like a FIFO queue.
  while (!WL1.empty()) {
    const ExplodedNode *N = WL1.back();
    WL1.pop_back();

    // Have we already visited this node?  If so, continue to the next one.
    if (Pass1.count(N))
      continue;

    // Otherwise, mark this node as visited.
    Pass1.insert(N);

    // If this is a root enqueue it to the second worklist.
    if (N->Preds.empty()) {
      WL2.push_back(N);
      continue;
    }

    // Visit our predecessors and enqueue them.
    for (ExplodedNode::pred_iterator I = N->Preds.begin(), E = N->Preds.end();
         I != E; ++I)
      WL1.push_back(*I);
  }

  // We didn't hit a root? Return with a null pointer for the new graph.
  if (WL2.empty())
    return 0;

  // Create an empty graph.
  ExplodedGraph* G = MakeEmptyGraph();

  // ===- Pass 2 (forward DFS to construct the new graph) -===
  while (!WL2.empty()) {
    const ExplodedNode *N = WL2.back();
    WL2.pop_back();

    // Skip this node if we have already processed it.
    if (Pass2.find(N) != Pass2.end())
      continue;

    // Create the corresponding node in the new graph and record the mapping
    // from the old node to the new node.
    ExplodedNode *NewN = G->getNode(N->getLocation(), N->State, N->isSink(), 0);
    Pass2[N] = NewN;

    // Also record the reverse mapping from the new node to the old node.
    if (InverseMap) (*InverseMap)[NewN] = N;

    // If this node is a root, designate it as such in the graph.
    if (N->Preds.empty())
      G->addRoot(NewN);

    // In the case that some of the intended predecessors of NewN have already
    // been created, we should hook them up as predecessors.

    // Walk through the predecessors of 'N' and hook up their corresponding
    // nodes in the new graph (if any) to the freshly created node.
    for (ExplodedNode::pred_iterator I = N->Preds.begin(), E = N->Preds.end();
         I != E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);
      if (PI == Pass2.end())
        continue;

      NewN->addPredecessor(PI->second, *G);
    }

    // In the case that some of the intended successors of NewN have already
    // been created, we should hook them up as successors.  Otherwise, enqueue
    // the new nodes from the original graph that should have nodes created
    // in the new graph.
    for (ExplodedNode::succ_iterator I = N->Succs.begin(), E = N->Succs.end();
         I != E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);
      if (PI != Pass2.end()) {
        PI->second->addPredecessor(NewN, *G);
        continue;
      }

      // Enqueue nodes to the worklist that were marked during pass 1.
      if (Pass1.count(*I))
        WL2.push_back(*I);
    }
  }

  return G;
}

void InterExplodedGraphMap::anchor() { }

ExplodedNode*
InterExplodedGraphMap::getMappedNode(const ExplodedNode *N) const {
  llvm::DenseMap<const ExplodedNode*, ExplodedNode*>::const_iterator I =
    M.find(N);

  return I == M.end() ? 0 : I->second;
}

