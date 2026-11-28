//===-- sanitizer_deadlock_detector.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// The deadlock detector maintains a directed graph of lock acquisitions.
// Each edge records its thread and the locks held before the acquisition.
// A cycle is reported only if its edges have distinct threads and compatible
// held-lock sets. Unlocking changes only the thread-local held locks.
// The caller's path buffer bounds the cycle length, as in the unlabeled graph.
//
// The detector can handle only a fixed amount of simultaneously live locks
// (a lock is alive if it has been locked at least once and has not been
// destroyed). When the maximal number of locks is reached the entire graph
// is flushed and the new lock epoch is started. The node ids from the old
// epochs can not be used with any of the detector methods except for
// nodeBelongsToCurrentEpoch().
//
// FIXME: this is work in progress, nothing really works yet.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_DEADLOCK_DETECTOR_H
#define SANITIZER_DEADLOCK_DETECTOR_H

#include "sanitizer_bvgraph.h"
#include "sanitizer_common.h"
#include "sanitizer_vector.h"

namespace __sanitizer {

// Thread-local state for DeadlockDetector.
// It contains the locks currently held by the owning thread.
template <class BV>
class DeadlockDetectorTLS {
 public:
  // No CTOR.
  void clear() {
    bv_.clear();
    write_bv_.clear();
    epoch_ = 0;
    n_recursive_locks = 0;
    n_all_locks_ = 0;
  }

  bool empty() const { return bv_.empty(); }

  void ensureCurrentEpoch(uptr current_epoch) {
    if (epoch_ == current_epoch) return;
    bv_.clear();
    write_bv_.clear();
    epoch_ = current_epoch;
    n_recursive_locks = 0;
    n_all_locks_ = 0;
  }

  uptr getEpoch() const { return epoch_; }

  // Returns true if this is the first (non-recursive) acquisition of this lock.
  bool addLock(uptr lock_id, uptr current_epoch, u32 stk, bool wlock = true) {
    CHECK_EQ(epoch_, current_epoch);
    if (!bv_.setBit(lock_id)) {
      // The lock is already held by this thread, it must be recursive.
      CHECK_LT(n_recursive_locks, ARRAY_SIZE(recursive_locks));
      recursive_locks[n_recursive_locks++] = lock_id;
      return false;
    }
    if (wlock)
      write_bv_.setBit(lock_id);
    CHECK_LT(n_all_locks_, ARRAY_SIZE(all_locks_with_contexts_));
    // lock_id < BV::kSize, can cast to a smaller int.
    u32 lock_id_short = static_cast<u32>(lock_id);
    LockWithContext l = {lock_id_short, stk};
    all_locks_with_contexts_[n_all_locks_++] = l;
    return true;
  }

  void removeLock(uptr lock_id) {
    if (n_recursive_locks) {
      for (sptr i = n_recursive_locks - 1; i >= 0; i--) {
        if (recursive_locks[i] == lock_id) {
          n_recursive_locks--;
          Swap(recursive_locks[i], recursive_locks[n_recursive_locks]);
          return;
        }
      }
    }
    if (!bv_.clearBit(lock_id))
      return;  // probably addLock happened before flush
    write_bv_.clearBit(lock_id);
    if (n_all_locks_) {
      for (sptr i = n_all_locks_ - 1; i >= 0; i--) {
        if (all_locks_with_contexts_[i].lock == static_cast<u32>(lock_id)) {
          Swap(all_locks_with_contexts_[i],
               all_locks_with_contexts_[n_all_locks_ - 1]);
          n_all_locks_--;
          break;
        }
      }
    }
  }

  u32 findLockContext(uptr lock_id) {
    for (uptr i = 0; i < n_all_locks_; i++)
      if (all_locks_with_contexts_[i].lock == static_cast<u32>(lock_id))
        return all_locks_with_contexts_[i].stk;
    return 0;
  }

  const BV &getLocks(uptr current_epoch) const {
    CHECK_EQ(epoch_, current_epoch);
    return bv_;
  }

  const BV& getWriteLocks(uptr current_epoch) const {
    CHECK_EQ(epoch_, current_epoch);
    return write_bv_;
  }

 private:
  BV bv_;
  BV write_bv_;
  uptr epoch_;
  uptr recursive_locks[64];
  uptr n_recursive_locks;
  struct LockWithContext {
    u32 lock;
    u32 stk;
  };
  LockWithContext all_locks_with_contexts_[512];
  uptr n_all_locks_;
};

// DeadlockDetector.
// For deadlock detection to work we need one global DeadlockDetector object
// and one DeadlockDetectorTLS object per evey thread.
// This class is not thread safe, all concurrent accesses should be guarded
// by an external lock.
// Most of the methods of this class are not thread-safe (i.e. should
// be protected by an external lock) unless explicitly told otherwise.
template <class BV>
class DeadlockDetector {
 public:
  typedef BV BitVector;

  struct Edge {
    uptr from;
    uptr to;
    u32 stk_from;
    u32 stk_to;
    int unique_tid;
    BV held_locks;
    BV write_locks;
  };

  uptr size() const { return g_.size(); }

  void clear() {
    current_epoch_ = 0;
    available_nodes_.clear();
    recycled_nodes_.clear();
    g_.clear();
    clearEdges();
  }

  // Allocate new deadlock detector node.
  // If we are out of available nodes first try to recycle some.
  // If there is nothing to recycle, flush the graph and increment the epoch.
  // Associate 'data' (opaque user's object) with the new node.
  uptr newNode(uptr data) {
    if (!available_nodes_.empty())
      return getAvailableNode(data);
    if (!recycled_nodes_.empty()) {
      for (uptr from = 0; from < size(); ++from) {
        auto& outgoing = edges_[from];
        for (uptr idx = outgoing.Size(); idx != 0; --idx) {
          Edge& edge = outgoing[idx - 1];
          if (recycled_nodes_.getBit(nodeToIndex(edge.to))) {
            if (idx != outgoing.Size())
              internal_memcpy(&edge, &outgoing[outgoing.Size() - 1],
                              sizeof(edge));
            outgoing.PopBack();
          } else {
            edge.held_locks.setDifference(recycled_nodes_);
            edge.write_locks.setDifference(recycled_nodes_);
          }
        }
      }
      CHECK(available_nodes_.empty());
      // removeEdgesFrom was called in removeNode.
      g_.removeEdgesTo(recycled_nodes_);
      available_nodes_.setUnion(recycled_nodes_);
      recycled_nodes_.clear();
      return getAvailableNode(data);
    }
    // We are out of vacant nodes. Flush and increment the current_epoch_.
    current_epoch_ += size();
    recycled_nodes_.clear();
    available_nodes_.setAll();
    g_.clear();
    clearEdges();
    return getAvailableNode(data);
  }

  // Get data associated with the node created by newNode().
  uptr getData(uptr node) const { return data_[nodeToIndex(node)]; }

  bool nodeBelongsToCurrentEpoch(uptr node) {
    return node && (node / size() * size()) == current_epoch_;
  }

  void removeNode(uptr node) {
    uptr idx = nodeToIndex(node);
    CHECK(!available_nodes_.getBit(idx));
    CHECK(recycled_nodes_.setBit(idx));
    g_.removeEdgesFrom(idx);
    edges_[idx].Reset();
  }

  void ensureCurrentEpoch(DeadlockDetectorTLS<BV> *dtls) {
    dtls->ensureCurrentEpoch(current_epoch_);
  }

  // Returns true if there is a cycle in the graph after this lock event.
  // Ideally should be called before the lock is acquired so that we can
  // report a deadlock before a real deadlock happens.
  bool onLockBefore(DeadlockDetectorTLS<BV> *dtls, uptr cur_node) {
    ensureCurrentEpoch(dtls);
    uptr cur_idx = nodeToIndex(cur_node);
    return g_.isReachable(cur_idx, dtls->getLocks(current_epoch_));
  }

  u32 findLockContext(DeadlockDetectorTLS<BV> *dtls, uptr node) {
    return dtls->findLockContext(nodeToIndex(node));
  }

  // Add cur_node to the set of locks held currently by dtls.
  void onLockAfter(DeadlockDetectorTLS<BV>* dtls, uptr cur_node, u32 stk = 0,
                   bool wlock = true) {
    ensureCurrentEpoch(dtls);
    uptr cur_idx = nodeToIndex(cur_node);
    dtls->addLock(cur_idx, current_epoch_, stk, wlock);
  }

  // Adds labeled edges from currently held locks to cur_node and returns
  // the number of new edges in the unlabeled graph.
  // Should be called before onLockAfter.
  uptr addEdges(DeadlockDetectorTLS<BV> *dtls, uptr cur_node, u32 stk,
                int unique_tid) {
    ensureCurrentEpoch(dtls);
    uptr cur_idx = nodeToIndex(cur_node);
    if (dtls->empty())
      return 0;
    uptr n_added_edges = 0;
    for (typename BV::Iterator it(dtls->getLocks(current_epoch_));
         it.hasNext();) {
      uptr from = it.next();
      if (g_.addEdge(from, cur_idx))
        ++n_added_edges;
      auto& outgoing = edges_[from];
      uptr idx = 0;
      for (; idx < outgoing.Size(); ++idx) {
        const Edge& edge = outgoing[idx];
        if (edge.to == cur_node && edge.unique_tid == unique_tid &&
            hasSameLocks(dtls, edge))
          break;
      }
      if (idx != outgoing.Size())
        continue;
      Edge* edge = outgoing.PushBack();
      edge->from = indexToNode(from);
      edge->to = cur_node;
      edge->stk_from = dtls->findLockContext(from);
      edge->stk_to = stk;
      edge->unique_tid = unique_tid;
      edge->held_locks.copyFrom(dtls->getLocks(current_epoch_));
      edge->write_locks.copyFrom(dtls->getWriteLocks(current_epoch_));
    }
    return n_added_edges;
  }

  uptr findGuardedCycle(DeadlockDetectorTLS<BV>* dtls, uptr cur_node,
                        int unique_tid, const Edge** path, uptr path_size) {
    ensureCurrentEpoch(dtls);
    uptr cur_idx = nodeToIndex(cur_node);
    for (uptr max_length = 2; max_length <= path_size; ++max_length) {
      for (typename BV::Iterator it(dtls->getLocks(current_epoch_));
           it.hasNext();) {
        const auto& outgoing = edges_[it.next()];
        for (uptr idx = 0; idx < outgoing.Size(); ++idx) {
          const Edge& closing = outgoing[idx];
          if (closing.to != cur_node || closing.unique_tid != unique_tid ||
              !hasSameLocks(dtls, closing))
            continue;
          path[0] = &closing;
          uptr length = findGuardedPath(cur_idx, path, 1, max_length);
          if (length) {
            for (uptr pos = 1; pos < length; ++pos)
              Swap(path[pos - 1], path[pos]);
            return length;
          }
        }
      }
    }
    return 0;
  }

  bool findEdge(uptr from_node, uptr to_node, u32* stk_from, u32* stk_to,
                int* unique_tid) {
    uptr from_idx = nodeToIndex(from_node);
    const auto& outgoing = edges_[from_idx];
    for (uptr idx = 0; idx < outgoing.Size(); ++idx) {
      const Edge& edge = outgoing[idx];
      if (edge.to == to_node) {
        *stk_from = edge.stk_from;
        *stk_to = edge.stk_to;
        *unique_tid = edge.unique_tid;
        return true;
      }
    }
    return false;
  }

  // Test-only function. Handles the before/after lock events,
  // returns true if there is a cycle.
  bool onLock(DeadlockDetectorTLS<BV> *dtls, uptr cur_node, u32 stk = 0) {
    ensureCurrentEpoch(dtls);
    bool is_reachable = !isHeld(dtls, cur_node) && onLockBefore(dtls, cur_node);
    addEdges(dtls, cur_node, stk, 0);
    onLockAfter(dtls, cur_node, stk);
    return is_reachable;
  }

  // Handles the try_lock event, returns false.
  // When a try_lock event happens (i.e. a try_lock call succeeds) we need
  // to add this lock to the currently held locks, but we should not try to
  // change the lock graph or to detect a cycle.  We may want to investigate
  // whether a more aggressive strategy is possible for try_lock.
  bool onTryLock(DeadlockDetectorTLS<BV> *dtls, uptr cur_node, u32 stk = 0) {
    ensureCurrentEpoch(dtls);
    uptr cur_idx = nodeToIndex(cur_node);
    dtls->addLock(cur_idx, current_epoch_, stk);
    return false;
  }

  // Returns true iff dtls is empty (no locks are currently held) and we can
  // add the node to the currently held locks w/o changing the global state.
  // This operation is thread-safe as it only touches the dtls.
  bool onFirstLock(DeadlockDetectorTLS<BV>* dtls, uptr node, u32 stk = 0,
                   bool wlock = true) {
    if (!dtls->empty())
      return false;
    if (dtls->getEpoch() && dtls->getEpoch() == nodeToEpoch(node)) {
      dtls->addLock(nodeToIndexUnchecked(node), nodeToEpoch(node), stk, wlock);
      return true;
    }
    return false;
  }

  // Finds a path between the lock 'cur_node' (currently not held in dtls)
  // and some currently held lock, returns the length of the path
  // or 0 on failure.
  uptr findPathToLock(DeadlockDetectorTLS<BV> *dtls, uptr cur_node, uptr *path,
                      uptr path_size) {
    tmp_bv_.copyFrom(dtls->getLocks(current_epoch_));
    uptr idx = nodeToIndex(cur_node);
    CHECK(!tmp_bv_.getBit(idx));
    uptr res = g_.findShortestPath(idx, tmp_bv_, path, path_size);
    for (uptr i = 0; i < res; i++)
      path[i] = indexToNode(path[i]);
    if (res)
      CHECK_EQ(path[0], cur_node);
    return res;
  }

  // Handle the unlock event.
  // This operation is thread-safe as it only touches the dtls.
  void onUnlock(DeadlockDetectorTLS<BV> *dtls, uptr node) {
    if (dtls->getEpoch() == nodeToEpoch(node))
      dtls->removeLock(nodeToIndexUnchecked(node));
  }

  bool isHeld(DeadlockDetectorTLS<BV> *dtls, uptr node) const {
    return dtls->getLocks(current_epoch_).getBit(nodeToIndex(node));
  }

  uptr testOnlyGetEpoch() const { return current_epoch_; }
  bool testOnlyHasEdge(uptr l1, uptr l2) {
    return g_.hasEdge(nodeToIndex(l1), nodeToIndex(l2));
  }
  // idx1 and idx2 are raw indices to g_, not lock IDs.
  bool testOnlyHasEdgeRaw(uptr idx1, uptr idx2) {
    return g_.hasEdge(idx1, idx2);
  }

  void Print() {
    for (uptr from = 0; from < size(); from++)
      for (uptr to = 0; to < size(); to++)
        if (g_.hasEdge(from, to))
          Printf("  %zx => %zx\n", from, to);
  }

 private:
  bool sameLocks(const BV& first, const BV& second) {
    tmp_bv_.copyFrom(first);
    tmp_bv_.setDifference(second);
    if (!tmp_bv_.empty())
      return false;
    tmp_bv_.copyFrom(second);
    tmp_bv_.setDifference(first);
    return tmp_bv_.empty();
  }

  bool hasSameLocks(DeadlockDetectorTLS<BV>* dtls, const Edge& edge) {
    return sameLocks(dtls->getLocks(current_epoch_), edge.held_locks) &&
           sameLocks(dtls->getWriteLocks(current_epoch_), edge.write_locks);
  }

  void clearEdges() {
    for (auto& outgoing : edges_) outgoing.Reset();
  }

  bool compatibleEdges(const Edge& first, const Edge& second) const {
    if (first.unique_tid >= 0 && first.unique_tid == second.unique_tid)
      return false;
    return !first.held_locks.intersectsWith(second.write_locks) &&
           !first.write_locks.intersectsWith(second.held_locks);
  }

  uptr findGuardedPath(uptr from, const Edge** path, uptr depth,
                       uptr path_size) {
    if (indexToNode(from) == path[0]->from)
      return depth;
    if (depth == path_size)
      return 0;
    const auto& outgoing = edges_[from];
    for (uptr idx = 0; idx < outgoing.Size(); ++idx) {
      const Edge& edge = outgoing[idx];
      bool compatible = true;
      for (uptr prev = 0; prev < depth; ++prev) {
        if (edge.to == path[prev]->to || !compatibleEdges(edge, *path[prev])) {
          compatible = false;
          break;
        }
      }
      if (!compatible)
        continue;
      path[depth] = &edge;
      uptr length =
          findGuardedPath(nodeToIndex(edge.to), path, depth + 1, path_size);
      if (length)
        return length;
    }
    return 0;
  }

  void check_idx(uptr idx) const { CHECK_LT(idx, size()); }

  void check_node(uptr node) const {
    CHECK_GE(node, size());
    CHECK_EQ(current_epoch_, nodeToEpoch(node));
  }

  uptr indexToNode(uptr idx) const {
    check_idx(idx);
    return idx + current_epoch_;
  }

  uptr nodeToIndexUnchecked(uptr node) const { return node % size(); }

  uptr nodeToIndex(uptr node) const {
    check_node(node);
    return nodeToIndexUnchecked(node);
  }

  uptr nodeToEpoch(uptr node) const { return node / size() * size(); }

  uptr getAvailableNode(uptr data) {
    uptr idx = available_nodes_.getAndClearFirstOne();
    data_[idx] = data;
    return indexToNode(idx);
  }

  uptr current_epoch_;
  BV available_nodes_;
  BV recycled_nodes_;
  BV tmp_bv_;
  BVGraph<BV> g_;
  uptr data_[BV::kSize];
  Vector<Edge> edges_[BV::kSize];
};

} // namespace __sanitizer

#endif // SANITIZER_DEADLOCK_DETECTOR_H
