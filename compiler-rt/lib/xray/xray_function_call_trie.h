//===-- xray_function_call_trie.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This file defines the interface for a function call trie.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_FUNCTION_CALL_TRIE_H
#define XRAY_FUNCTION_CALL_TRIE_H

#include "xray_defs.h"
#include "xray_profiling_flags.h"
#include "xray_segmented_array.h"
#include <memory> // For placement new.
#include <utility>

namespace __xray {

/// A FunctionCallTrie represents the stack traces of XRay instrumented
/// functions that we've encountered, where a node corresponds to a function and
/// the path from the root to the node its stack trace. Each node in the trie
/// will contain some useful values, including:
///
///   * The cumulative amount of time spent in this particular node/stack.
///   * The number of times this stack has appeared.
///   * A histogram of latencies for that particular node.
///
/// Each node in the trie will also contain a list of callees, represented using
/// a Array<NodeIdPair> -- each NodeIdPair instance will contain the function
/// ID of the callee, and a pointer to the node.
///
/// If we visualise this data structure, we'll find the following potential
/// representation:
///
///   [function id node] -> [callees] [cumulative time]
///                         [call counter] [latency histogram]
///
/// As an example, when we have a function in this pseudocode:
///
///   func f(N) {
///     g()
///     h()
///     for i := 1..N { j() }
///   }
///
/// We may end up with a trie of the following form:
///
///   f -> [ g, h, j ] [...] [1] [...]
///   g -> [ ... ] [...] [1] [...]
///   h -> [ ... ] [...] [1] [...]
///   j -> [ ... ] [...] [N] [...]
///
/// If for instance the function g() called j() like so:
///
///   func g() {
///     for i := 1..10 { j() }
///   }
///
/// We'll find the following updated trie:
///
///   f -> [ g, h, j ] [...] [1] [...]
///   g -> [ j' ] [...] [1] [...]
///   h -> [ ... ] [...] [1] [...]
///   j -> [ ... ] [...] [N] [...]
///   j' -> [ ... ] [...] [10] [...]
///
/// Note that we'll have a new node representing the path `f -> g -> j'` with
/// isolated data. This isolation gives us a means of representing the stack
/// traces as a path, as opposed to a key in a table. The alternative
/// implementation here would be to use a separate table for the path, and use
/// hashes of the path as an identifier to accumulate the information. We've
/// moved away from this approach as it takes a lot of time to compute the hash
/// every time we need to update a function's call information as we're handling
/// the entry and exit events.
///
/// This approach allows us to maintain a shadow stack, which represents the
/// currently executing path, and on function exits quickly compute the amount
/// of time elapsed from the entry, then update the counters for the node
/// already represented in the trie. This necessitates an efficient
/// representation of the various data structures (the list of callees must be
/// cache-aware and efficient to look up, and the histogram must be compact and
/// quick to update) to enable us to keep the overheads of this implementation
/// to the minimum.
class FunctionCallTrie {
public:
  struct Node;

  // We use a NodeIdPair type instead of a std::pair<...> to not rely on the
  // standard library types in this header.
  struct NodeIdPair {
    Node *NodePtr;
    int32_t FId;

    // Constructor for inplace-construction.
    NodeIdPair(Node *N, int32_t F) : NodePtr(N), FId(F) {}
  };

  using NodeIdPairArray = Array<NodeIdPair>;
  using NodeIdPairAllocatorType = NodeIdPairArray::AllocatorType;

  // A Node in the FunctionCallTrie gives us a list of callees, the cumulative
  // number of times this node actually appeared, the cumulative amount of time
  // for this particular node including its children call times, and just the
  // local time spent on this node. Each Node will have the ID of the XRay
  // instrumented function that it is associated to.
  struct Node {
    Node *Parent;
    NodeIdPairArray Callees;
    int64_t CallCount;
    int64_t CumulativeLocalTime; // Typically in TSC deltas, not wall-time.
    int32_t FId;

    // We add a constructor here to allow us to inplace-construct through
    // Array<...>'s AppendEmplace.
    Node(Node *P, NodeIdPairAllocatorType &A, int64_t CC, int64_t CLT,
         int32_t F) XRAY_NEVER_INSTRUMENT : Parent(P),
                                            Callees(A),
                                            CallCount(CC),
                                            CumulativeLocalTime(CLT),
                                            FId(F) {}

    // TODO: Include the compact histogram.
  };

private:
  struct ShadowStackEntry {
    uint64_t EntryTSC;
    Node *NodePtr;

    // We add a constructor here to allow us to inplace-construct through
    // Array<...>'s AppendEmplace.
    ShadowStackEntry(uint64_t T, Node *N) XRAY_NEVER_INSTRUMENT : EntryTSC{T},
                                                                  NodePtr{N} {}
  };

  using NodeArray = Array<Node>;
  using RootArray = Array<Node *>;
  using ShadowStackArray = Array<ShadowStackEntry>;

public:
  // We collate the allocators we need into a single struct, as a convenience to
  // allow us to initialize these as a group.
  struct Allocators {
    using NodeAllocatorType = NodeArray::AllocatorType;
    using RootAllocatorType = RootArray::AllocatorType;
    using ShadowStackAllocatorType = ShadowStackArray::AllocatorType;

    NodeAllocatorType *NodeAllocator = nullptr;
    RootAllocatorType *RootAllocator = nullptr;
    ShadowStackAllocatorType *ShadowStackAllocator = nullptr;
    NodeIdPairAllocatorType *NodeIdPairAllocator = nullptr;

    Allocators() {}
    Allocators(const Allocators &) = delete;
    Allocators &operator=(const Allocators &) = delete;

    Allocators(Allocators &&O) XRAY_NEVER_INSTRUMENT
        : NodeAllocator(O.NodeAllocator),
          RootAllocator(O.RootAllocator),
          ShadowStackAllocator(O.ShadowStackAllocator),
          NodeIdPairAllocator(O.NodeIdPairAllocator) {
      O.NodeAllocator = nullptr;
      O.RootAllocator = nullptr;
      O.ShadowStackAllocator = nullptr;
      O.NodeIdPairAllocator = nullptr;
    }

    Allocators &operator=(Allocators &&O) XRAY_NEVER_INSTRUMENT {
      {
        auto Tmp = O.NodeAllocator;
        O.NodeAllocator = this->NodeAllocator;
        this->NodeAllocator = Tmp;
      }
      {
        auto Tmp = O.RootAllocator;
        O.RootAllocator = this->RootAllocator;
        this->RootAllocator = Tmp;
      }
      {
        auto Tmp = O.ShadowStackAllocator;
        O.ShadowStackAllocator = this->ShadowStackAllocator;
        this->ShadowStackAllocator = Tmp;
      }
      {
        auto Tmp = O.NodeIdPairAllocator;
        O.NodeIdPairAllocator = this->NodeIdPairAllocator;
        this->NodeIdPairAllocator = Tmp;
      }
      return *this;
    }

    ~Allocators() XRAY_NEVER_INSTRUMENT {
      // Note that we cannot use delete on these pointers, as they need to be
      // returned to the sanitizer_common library's internal memory tracking
      // system.
      if (NodeAllocator != nullptr) {
        NodeAllocator->~NodeAllocatorType();
        deallocate(NodeAllocator);
        NodeAllocator = nullptr;
      }
      if (RootAllocator != nullptr) {
        RootAllocator->~RootAllocatorType();
        deallocate(RootAllocator);
        RootAllocator = nullptr;
      }
      if (ShadowStackAllocator != nullptr) {
        ShadowStackAllocator->~ShadowStackAllocatorType();
        deallocate(ShadowStackAllocator);
        ShadowStackAllocator = nullptr;
      }
      if (NodeIdPairAllocator != nullptr) {
        NodeIdPairAllocator->~NodeIdPairAllocatorType();
        deallocate(NodeIdPairAllocator);
        NodeIdPairAllocator = nullptr;
      }
    }
  };

  // TODO: Support configuration of options through the arguments.
  static Allocators InitAllocators() XRAY_NEVER_INSTRUMENT {
    return InitAllocatorsCustom(profilingFlags()->per_thread_allocator_max);
  }

  static Allocators InitAllocatorsCustom(uptr Max) XRAY_NEVER_INSTRUMENT {
    Allocators A;
    auto NodeAllocator = allocate<Allocators::NodeAllocatorType>();
    new (NodeAllocator) Allocators::NodeAllocatorType(Max);
    A.NodeAllocator = NodeAllocator;

    auto RootAllocator = allocate<Allocators::RootAllocatorType>();
    new (RootAllocator) Allocators::RootAllocatorType(Max);
    A.RootAllocator = RootAllocator;

    auto ShadowStackAllocator =
        allocate<Allocators::ShadowStackAllocatorType>();
    new (ShadowStackAllocator) Allocators::ShadowStackAllocatorType(Max);
    A.ShadowStackAllocator = ShadowStackAllocator;

    auto NodeIdPairAllocator = allocate<NodeIdPairAllocatorType>();
    new (NodeIdPairAllocator) NodeIdPairAllocatorType(Max);
    A.NodeIdPairAllocator = NodeIdPairAllocator;
    return A;
  }

private:
  NodeArray Nodes;
  RootArray Roots;
  ShadowStackArray ShadowStack;
  NodeIdPairAllocatorType *NodeIdPairAllocator = nullptr;

public:
  explicit FunctionCallTrie(const Allocators &A) XRAY_NEVER_INSTRUMENT
      : Nodes(*A.NodeAllocator),
        Roots(*A.RootAllocator),
        ShadowStack(*A.ShadowStackAllocator),
        NodeIdPairAllocator(A.NodeIdPairAllocator) {}

  void enterFunction(const int32_t FId, uint64_t TSC) XRAY_NEVER_INSTRUMENT {
    DCHECK_NE(FId, 0);
    // This function primarily deals with ensuring that the ShadowStack is
    // consistent and ready for when an exit event is encountered.
    if (UNLIKELY(ShadowStack.empty())) {
      auto NewRoot =
          Nodes.AppendEmplace(nullptr, *NodeIdPairAllocator, 0, 0, FId);
      if (UNLIKELY(NewRoot == nullptr))
        return;
      Roots.Append(NewRoot);
      ShadowStack.AppendEmplace(TSC, NewRoot);
      return;
    }

    auto &Top = ShadowStack.back();
    auto TopNode = Top.NodePtr;
    DCHECK_NE(TopNode, nullptr);

    // If we've seen this callee before, then we just access that node and place
    // that on the top of the stack.
    auto Callee = TopNode->Callees.find_element(
        [FId](const NodeIdPair &NR) { return NR.FId == FId; });
    if (Callee != nullptr) {
      CHECK_NE(Callee->NodePtr, nullptr);
      ShadowStack.AppendEmplace(TSC, Callee->NodePtr);
      return;
    }

    // This means we've never seen this stack before, create a new node here.
    auto NewNode =
        Nodes.AppendEmplace(TopNode, *NodeIdPairAllocator, 0, 0, FId);
    if (UNLIKELY(NewNode == nullptr))
      return;
    DCHECK_NE(NewNode, nullptr);
    TopNode->Callees.AppendEmplace(NewNode, FId);
    ShadowStack.AppendEmplace(TSC, NewNode);
    DCHECK_NE(ShadowStack.back().NodePtr, nullptr);
    return;
  }

  void exitFunction(int32_t FId, uint64_t TSC) XRAY_NEVER_INSTRUMENT {
    // When we exit a function, we look up the ShadowStack to see whether we've
    // entered this function before. We do as little processing here as we can,
    // since most of the hard work would have already been done at function
    // entry.
    uint64_t CumulativeTreeTime = 0;
    while (!ShadowStack.empty()) {
      const auto &Top = ShadowStack.back();
      auto TopNode = Top.NodePtr;
      DCHECK_NE(TopNode, nullptr);
      auto LocalTime = TSC - Top.EntryTSC;
      TopNode->CallCount++;
      TopNode->CumulativeLocalTime += LocalTime - CumulativeTreeTime;
      CumulativeTreeTime += LocalTime;
      ShadowStack.trim(1);

      // TODO: Update the histogram for the node.
      if (TopNode->FId == FId)
        break;
    }
  }

  const RootArray &getRoots() const XRAY_NEVER_INSTRUMENT { return Roots; }

  // The deepCopyInto operation will update the provided FunctionCallTrie by
  // re-creating the contents of this particular FunctionCallTrie in the other
  // FunctionCallTrie. It will do this using a Depth First Traversal from the
  // roots, and while doing so recreating the traversal in the provided
  // FunctionCallTrie.
  //
  // This operation will *not* destroy the state in `O`, and thus may cause some
  // duplicate entries in `O` if it is not empty.
  //
  // This function is *not* thread-safe, and may require external
  // synchronisation of both "this" and |O|.
  //
  // This function must *not* be called with a non-empty FunctionCallTrie |O|.
  void deepCopyInto(FunctionCallTrie &O) const XRAY_NEVER_INSTRUMENT {
    DCHECK(O.getRoots().empty());

    // We then push the root into a stack, to use as the parent marker for new
    // nodes we push in as we're traversing depth-first down the call tree.
    struct NodeAndParent {
      FunctionCallTrie::Node *Node;
      FunctionCallTrie::Node *NewNode;
    };
    using Stack = Array<NodeAndParent>;

    typename Stack::AllocatorType StackAllocator(
        profilingFlags()->stack_allocator_max);
    Stack DFSStack(StackAllocator);

    for (const auto Root : getRoots()) {
      // Add a node in O for this root.
      auto NewRoot = O.Nodes.AppendEmplace(
          nullptr, *O.NodeIdPairAllocator, Root->CallCount,
          Root->CumulativeLocalTime, Root->FId);

      // Because we cannot allocate more memory we should bail out right away.
      if (UNLIKELY(NewRoot == nullptr))
        return;

      O.Roots.Append(NewRoot);

      // TODO: Figure out what to do if we fail to allocate any more stack
      // space. Maybe warn or report once?
      DFSStack.AppendEmplace(Root, NewRoot);
      while (!DFSStack.empty()) {
        NodeAndParent NP = DFSStack.back();
        DCHECK_NE(NP.Node, nullptr);
        DCHECK_NE(NP.NewNode, nullptr);
        DFSStack.trim(1);
        for (const auto Callee : NP.Node->Callees) {
          auto NewNode = O.Nodes.AppendEmplace(
              NP.NewNode, *O.NodeIdPairAllocator, Callee.NodePtr->CallCount,
              Callee.NodePtr->CumulativeLocalTime, Callee.FId);
          if (UNLIKELY(NewNode == nullptr))
            return;
          NP.NewNode->Callees.AppendEmplace(NewNode, Callee.FId);
          DFSStack.AppendEmplace(Callee.NodePtr, NewNode);
        }
      }
    }
  }

  // The mergeInto operation will update the provided FunctionCallTrie by
  // traversing the current trie's roots and updating (i.e. merging) the data in
  // the nodes with the data in the target's nodes. If the node doesn't exist in
  // the provided trie, we add a new one in the right position, and inherit the
  // data from the original (current) trie, along with all its callees.
  //
  // This function is *not* thread-safe, and may require external
  // synchronisation of both "this" and |O|.
  void mergeInto(FunctionCallTrie &O) const XRAY_NEVER_INSTRUMENT {
    struct NodeAndTarget {
      FunctionCallTrie::Node *OrigNode;
      FunctionCallTrie::Node *TargetNode;
    };
    using Stack = Array<NodeAndTarget>;
    typename Stack::AllocatorType StackAllocator(
        profilingFlags()->stack_allocator_max);
    Stack DFSStack(StackAllocator);

    for (const auto Root : getRoots()) {
      Node *TargetRoot = nullptr;
      auto R = O.Roots.find_element(
          [&](const Node *Node) { return Node->FId == Root->FId; });
      if (R == nullptr) {
        TargetRoot = O.Nodes.AppendEmplace(nullptr, *O.NodeIdPairAllocator, 0,
                                           0, Root->FId);
        if (UNLIKELY(TargetRoot == nullptr))
          return;

        O.Roots.Append(TargetRoot);
      } else {
        TargetRoot = *R;
      }

      DFSStack.Append(NodeAndTarget{Root, TargetRoot});
      while (!DFSStack.empty()) {
        NodeAndTarget NT = DFSStack.back();
        DCHECK_NE(NT.OrigNode, nullptr);
        DCHECK_NE(NT.TargetNode, nullptr);
        DFSStack.trim(1);
        // TODO: Update the histogram as well when we have it ready.
        NT.TargetNode->CallCount += NT.OrigNode->CallCount;
        NT.TargetNode->CumulativeLocalTime += NT.OrigNode->CumulativeLocalTime;
        for (const auto Callee : NT.OrigNode->Callees) {
          auto TargetCallee = NT.TargetNode->Callees.find_element(
              [&](const FunctionCallTrie::NodeIdPair &C) {
                return C.FId == Callee.FId;
              });
          if (TargetCallee == nullptr) {
            auto NewTargetNode = O.Nodes.AppendEmplace(
                NT.TargetNode, *O.NodeIdPairAllocator, 0, 0, Callee.FId);

            if (UNLIKELY(NewTargetNode == nullptr))
              return;

            TargetCallee =
                NT.TargetNode->Callees.AppendEmplace(NewTargetNode, Callee.FId);
          }
          DFSStack.AppendEmplace(Callee.NodePtr, TargetCallee->NodePtr);
        }
      }
    }
  }
};

} // namespace __xray

#endif // XRAY_FUNCTION_CALL_TRIE_H
