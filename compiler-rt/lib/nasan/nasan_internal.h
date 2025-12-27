//===-- nasan_internal.h - NoAliasSanitizer Internal Definitions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//
//
// Internal data structures and definitions for NoAliasSanitizer.
// Uses sanitizer_common data structures for consistency with other sanitizers.
//
//===----------------------------------------------------------------------===//

#ifndef NASAN_INTERNAL_H
#define NASAN_INTERNAL_H

#include "nasan.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_dense_map_info.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __nasan {

using namespace __sanitizer;

// Source location information
struct SourceLocation {
  const char *file;
  int line;
};

// Provenance metadata
struct ProvenanceInfo {
  ProvenanceID id;
  ProvenanceID based_on;   // 0 if root provenance
  void *noalias_param;     // Original noalias parameter
  const char *function_name;
  SourceLocation location;
  bool is_allocation;      // true for malloc/new
};

// Per-pointer shadow metadata (8-byte aligned)
struct PointerShadow {
  ProvenanceID primary_prov;  // 8 bytes
};

//===----------------------------------------------------------------------===//
// DenseSet - A set implementation using DenseMap<T, bool>
//===----------------------------------------------------------------------===//

template <typename T, typename KeyInfoT = DenseMapInfo<T>>
class DenseSet {
  DenseMap<T, bool, KeyInfoT> Map;

public:
  DenseSet() = default;

  void insert(const T &val) { Map[val] = true; }

  bool contains(const T &val) const { return Map.contains(val); }

  bool erase(const T &val) { return Map.erase(val); }

  void clear() { Map.clear(); }

  unsigned size() const { return Map.size(); }

  bool empty() const { return Map.empty(); }

  // Iterator support using forEach
  template <class Fn>
  void forEach(Fn fn) const {
    Map.forEach([&](const detail::DenseMapPair<T, bool> &KV) {
      fn(KV.first);
      return true;
    });
  }

  // Simple iterator for range-based for loops
  class Iterator {
  public:
    using MapIterator =
        typename DenseMap<T, bool, KeyInfoT>::value_type *;

    Iterator(const DenseSet *set, bool end) : set_(set), end_(end) {
      if (!end_) {
        // Collect all elements into a vector for iteration
        set_->forEach([this](const T &val) { elements_.push_back(val); });
        idx_ = 0;
      }
    }

    const T &operator*() const { return elements_[idx_]; }

    Iterator &operator++() {
      idx_++;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      if (end_ && other.end_)
        return false;
      if (end_)
        return other.idx_ < other.elements_.size();
      if (other.end_)
        return idx_ < elements_.size();
      return idx_ != other.idx_;
    }

  private:
    const DenseSet *set_;
    bool end_;
    InternalMmapVector<T> elements_;
    uptr idx_ = 0;
  };

  Iterator begin() const { return Iterator(this, false); }
  Iterator end() const { return Iterator(this, true); }
};

//===----------------------------------------------------------------------===//
// Data structures using DenseMap/DenseSet
//===----------------------------------------------------------------------===//

// Per-memory-location access tracking
struct MemoryAccessInfo {
  DenseSet<ProvenanceID> provenances;
  u32 scope_depth;
};

// Global state (per-thread)
struct NASanThreadState {
  // Memory access tracking (key = memory address)
  DenseMap<void *, MemoryAccessInfo> memory_access;

  // Provenance metadata
  DenseMap<ProvenanceID, ProvenanceInfo> provenance_info;

  // Track merged provenances (PHI nodes)
  DenseMap<void *, DenseSet<ProvenanceID>> merged_provenance;

  // Track pointers stored to memory (for load propagation)
  DenseMap<void *, ProvenanceID> stored_pointer_provenance;

  ProvenanceID next_prov_id;

  NASanThreadState() : next_prov_id(1) {}
};

// Get per-thread state
NASanThreadState *get_thread_state();

// Shadow memory functions
void init_shadow_memory();
bool is_shadow_initialized();
PointerShadow *get_shadow(void *ptr);

// Provenance checking
bool is_based_on(ProvenanceID prov1, ProvenanceID prov2);
bool are_compatible(ProvenanceID prov1, ProvenanceID prov2);
void get_all_provenances(void *ptr, InternalMmapVector<ProvenanceID> *result);

// Violation reporting
void report_violation(void *addr,
                      const InternalMmapVector<ProvenanceID> &accessing_provs,
                      const DenseSet<ProvenanceID> &existing_provs);

// Statistics (thread-safe using atomics)
struct NASanStats {
  atomic_uint64_t total_provs;
  atomic_uint64_t total_checks;
  atomic_uint64_t violations;
  atomic_uint64_t shadow_bytes;

  NASanStats() {
    atomic_store(&total_provs, 0, memory_order_relaxed);
    atomic_store(&total_checks, 0, memory_order_relaxed);
    atomic_store(&violations, 0, memory_order_relaxed);
    atomic_store(&shadow_bytes, 0, memory_order_relaxed);
  }
};

extern NASanStats g_stats;

} // namespace __nasan

#endif // NASAN_INTERNAL_H
