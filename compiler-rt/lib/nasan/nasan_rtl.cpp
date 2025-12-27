//===-- nasan_rtl.cpp - NoAliasSanitizer Runtime Library -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main runtime implementation for NoAliasSanitizer.
//
//===----------------------------------------------------------------------===//

#include "nasan_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

// Placement new declaration (can't include <new> due to -nostdinc++)
inline void *operator new(__SIZE_TYPE__, void *__p) noexcept { return __p; }

namespace __nasan {

using namespace __sanitizer;

// Per-thread state
static thread_local NASanThreadState *g_nasan_state = nullptr;

// Global statistics
NASanStats g_stats;

NASanThreadState *get_thread_state() {
  if (!g_nasan_state) {
    g_nasan_state = (NASanThreadState *)InternalAlloc(sizeof(NASanThreadState));
    new (g_nasan_state) NASanThreadState();
  }
  return g_nasan_state;
}

// Check if prov1 is based on prov2 (transitively)
bool is_based_on(ProvenanceID prov1, ProvenanceID prov2) {
  NASanThreadState *state = get_thread_state();

  ProvenanceID current = prov1;
  int depth = 0;
  while (current != 0 && depth < 100) {  // Prevent infinite loops
    if (current == prov2) return true;
    auto *entry = state->provenance_info.find(current);
    if (!entry) break;
    current = entry->second.based_on;
    depth++;
  }
  return false;
}

// Check if two provenances are compatible (same chain)
bool are_compatible(ProvenanceID prov1, ProvenanceID prov2) {
  if (prov1 == prov2) return true;  // Same provenance or both zero
  // Provenance 0 means "unknown/not from noalias param" - it's compatible with
  // everything since we only care about conflicts between noalias-derived pointers
  if (prov1 == 0 || prov2 == 0) return true;
  return is_based_on(prov1, prov2) || is_based_on(prov2, prov1);
}

// Get all provenances for a pointer (handles PHI merges)
void get_all_provenances(void *ptr, InternalMmapVector<ProvenanceID> *result) {
  NASanThreadState *state = get_thread_state();
  result->clear();

  // Check for merged provenance first
  auto *merged_entry = state->merged_provenance.find(ptr);
  if (merged_entry) {
    merged_entry->second.forEach([result](const ProvenanceID &prov) {
      result->push_back(prov);
    });
    return;
  }

  // Single provenance case
  ProvenanceID prov = __nasan_get_pointer_provenance(ptr);
  result->push_back(prov);
}

} // namespace __nasan

using namespace __nasan;

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_init() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  init_shadow_memory();

  // Check environment variables
  const char *options = GetEnv("NASAN_OPTIONS");
  if (options) {
    // Parse options (for future enhancement)
  }
}

// Ensure initialization happens automatically
__attribute__((constructor))
static void nasan_init_constructor() {
  __nasan_init();
}

SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_create_provenance(void *param, const char *func_name,
                                       const char *file, int line) {
  NASanThreadState *state = get_thread_state();
  ProvenanceID prov_id = state->next_prov_id++;

  ProvenanceInfo info;
  info.id = prov_id;
  info.based_on = 0;  // Parameters are independent roots, not based on each other
  info.noalias_param = param;
  info.function_name = func_name;
  info.location.file = file;
  info.location.line = line;
  info.is_allocation = false;

  state->provenance_info[prov_id] = info;

  atomic_fetch_add(&g_stats.total_provs, 1, memory_order_relaxed);

  return prov_id;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_destroy_provenance(ProvenanceID prov) {
  if (prov == 0) return;

  NASanThreadState *state = get_thread_state();

  // Remove this provenance from all memory_access entries
  // Collect addresses that become empty after removal
  InternalMmapVector<void *> addrs_to_remove;

  state->memory_access.forEach([&](detail::DenseMapPair<void *, MemoryAccessInfo> &entry) {
    entry.second.provenances.erase(prov);
    if (entry.second.provenances.size() == 0) {
      addrs_to_remove.push_back(entry.first);
    }
    return true;
  });

  // Remove empty entries
  for (uptr i = 0; i < addrs_to_remove.size(); i++) {
    state->memory_access.erase(addrs_to_remove[i]);
  }

  // Remove from merged_provenance entries
  InternalMmapVector<void *> merged_to_remove;
  state->merged_provenance.forEach([&](detail::DenseMapPair<void *, DenseSet<ProvenanceID>> &entry) {
    entry.second.erase(prov);
    if (entry.second.size() == 0) {
      merged_to_remove.push_back(entry.first);
    }
    return true;
  });

  for (uptr i = 0; i < merged_to_remove.size(); i++) {
    state->merged_provenance.erase(merged_to_remove[i]);
  }

  // Remove from provenance_info
  state->provenance_info.erase(prov);
}

SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_create_allocation_provenance(void *ptr, uptr size) {
  NASanThreadState *state = get_thread_state();
  ProvenanceID prov_id = state->next_prov_id++;

  ProvenanceInfo info;
  info.id = prov_id;
  info.based_on = 0;  // Allocations are root provenances
  info.noalias_param = ptr;
  info.function_name = "allocation";
  info.location.file = "<runtime>";
  info.location.line = 0;
  info.is_allocation = true;

  state->provenance_info[prov_id] = info;
  __nasan_set_pointer_provenance(ptr, prov_id);

  atomic_fetch_add(&g_stats.total_provs, 1, memory_order_relaxed);

  return prov_id;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_destroy_allocation_provenance(void *ptr) {
  ProvenanceID prov = __nasan_get_pointer_provenance(ptr);

  if (prov != 0) {
    // Clean up provenance info
    __nasan_destroy_provenance(prov);
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_record_pointer_store(void *addr, ProvenanceID prov) {
  NASanThreadState *state = get_thread_state();
  // Store the provenance directly (passed from compile-time tracking)
  state->stored_pointer_provenance[addr] = prov;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_propagate_through_load(void *dst_ptr, void *src_addr) {
  NASanThreadState *state = get_thread_state();
  auto *entry = state->stored_pointer_provenance.find(src_addr);

  if (entry) {
    __nasan_set_pointer_provenance(dst_ptr, entry->second);
  } else {
    // No stored provenance, treat as provenance 0
    __nasan_set_pointer_provenance(dst_ptr, 0);
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_merge_provenance(void *dst_ptr, void **src_ptrs, uptr count) {
  NASanThreadState *state = get_thread_state();
  DenseSet<ProvenanceID> provs;

  for (uptr i = 0; i < count; ++i) {
    ProvenanceID prov = __nasan_get_pointer_provenance(src_ptrs[i]);
    if (prov != 0) provs.insert(prov);
  }

  if (provs.size() == 0) {
    // No provenance from any source
    __nasan_set_pointer_provenance(dst_ptr, 0);
  } else if (provs.size() == 1) {
    // Single provenance - use fast path
    ProvenanceID single_prov = 0;
    provs.forEach([&single_prov](const ProvenanceID &p) { single_prov = p; });
    __nasan_set_pointer_provenance(dst_ptr, single_prov);
  } else {
    // Multiple provenances - store in separate map
    // Use first provenance as primary for shadow memory
    ProvenanceID first_prov = 0;
    provs.forEach([&first_prov](const ProvenanceID &p) {
      if (first_prov == 0) first_prov = p;
    });
    __nasan_set_pointer_provenance(dst_ptr, first_prov);

    // Copy elements to the merged provenance map
    DenseSet<ProvenanceID> &dst_set = state->merged_provenance[dst_ptr];
    dst_set.clear();
    provs.forEach([&dst_set](const ProvenanceID &prov) {
      dst_set.insert(prov);
    });
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_check_store(u64 addr_int, u64 size, ProvenanceID prov) {
  NASanThreadState *state = get_thread_state();
  void *addr = reinterpret_cast<void *>(addr_int);

  atomic_fetch_add(&g_stats.total_checks, 1, memory_order_relaxed);

  // Provenance 0 means no tracking needed (not from noalias)
  if (prov == 0)
    return;

  // Check each byte in the access range
  for (u64 offset = 0; offset < size; ++offset) {
    void *byte_addr = static_cast<char *>(addr) + offset;
    MemoryAccessInfo &access_info = state->memory_access[byte_addr];

    // If first access, just record it
    if (access_info.provenances.size() == 0) {
      access_info.provenances.insert(prov);
      continue;
    }

    // Check compatibility with existing accesses
    bool compatible = false;
    access_info.provenances.forEach([&](const ProvenanceID &existing_prov) {
      if (are_compatible(prov, existing_prov)) {
        compatible = true;
      }
    });

    if (compatible) {
      access_info.provenances.insert(prov);
    } else {
      // VIOLATION!
      atomic_fetch_add(&g_stats.violations, 1, memory_order_relaxed);
      InternalMmapVector<ProvenanceID> accessing_provs;
      accessing_provs.push_back(prov);
      report_violation(byte_addr, accessing_provs, access_info.provenances);
    }
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_check_load(u64 addr_int, u64 size, ProvenanceID prov) {
  // For loads, we validate the same way as stores
  // (noalias applies to both read and write accesses)
  __nasan_check_store(addr_int, size, prov);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_handle_exception_cleanup(ProvenanceID prov) {
  // Called from landing pads to clean up provenance on exception
  __nasan_destroy_provenance(prov);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_function_entry() {
  // Reserved for future use
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_function_exit() {
  // Reserved for future use
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_print_pointer_info(void *ptr) {
  Printf("Pointer %p provenance info:\n", ptr);
  InternalMmapVector<ProvenanceID> provs;
  get_all_provenances(ptr, &provs);

  for (uptr i = 0; i < provs.size(); i++) {
    Printf("  Provenance ID: %llu\n", (unsigned long long)provs[i]);
    // More details could be added here
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_dump_state() {
  auto* state = get_thread_state();

  Printf("\n=== NASan State Dump ===\n");
  Printf("Active provenances: %u\n", state->provenance_info.size());
  Printf("Tracked memory locations: %u\n", state->memory_access.size());
  Printf("Merged provenances: %u\n", state->merged_provenance.size());
  Printf("\n=== Statistics ===\n");
  Printf("Total provenances created: %llu\n",
         (unsigned long long)atomic_load(&g_stats.total_provs, memory_order_relaxed));
  Printf("Total checks: %llu\n",
         (unsigned long long)atomic_load(&g_stats.total_checks, memory_order_relaxed));
  Printf("Violations detected: %llu\n",
         (unsigned long long)atomic_load(&g_stats.violations, memory_order_relaxed));
  Printf("\n");
}

} // extern "C"
