//===-- pointer_tracking_runtime.cpp - Pointer Tracking Runtime ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a runtime for tracking pointer allocations and usage
// patterns. It replaces real pointers with fake pointers encoding allocation
// IDs, maintains a global table tracking metadata (read/write status, timing),
// and reports statistics on allocation usage patterns.
//
//===----------------------------------------------------------------------===//

#include "../instrumentor_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// Use high-resolution clock for timing measurements
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds;

// Metadata for each allocation
struct AllocationMetadata {
  void *real_pointer;        // The actual pointer value
  uint64_t size;             // Size of the allocation in bytes
  bool was_read;             // Has this allocation been read?
  bool was_written;          // Has this allocation been written?
  TimePoint allocation_time; // When was this allocated?
  TimePoint first_use_time;  // When was it first used?
  TimePoint last_use_time;   // When was it last used?
  bool has_been_used;        // Has it been used at all?
  int32_t alloc_id;          // The allocation ID
  bool is_global;            // Is this a global variable?
  char name[256];            // Name (for globals)
};

// Global table mapping allocation IDs to metadata
// Use pointers to avoid destruction order issues with the destructor
static std::unordered_map<uint64_t, AllocationMetadata> *g_allocation_table =
    nullptr;

// Stack tracking which allocations belong to each function scope
// Each element is a vector of allocation IDs created in that function
static std::vector<std::vector<uint64_t>> *g_function_alloc_stack = nullptr;

// Statistics for timing analysis
struct TimingRecord {
  int32_t alloc_id;
  Duration duration;
};

static std::vector<TimingRecord> *g_unused_timers =
    nullptr; // Time from last use to deallocation
static std::vector<TimingRecord> *g_first_use_timers =
    nullptr; // Time from allocation to first use

// Accumulated statistics
static std::atomic<uint64_t> *g_total_allocations = nullptr;
static std::atomic<uint64_t> *g_read_only_count = nullptr;
static std::atomic<uint64_t> *g_write_only_count = nullptr;
static std::atomic<uint64_t> *g_read_write_count = nullptr;
static std::atomic<uint64_t> *g_unused_count = nullptr;

// Initialize globals on first use
static void init_globals() {
  static bool initialized = false;
  if (!initialized) {
    g_allocation_table = new std::unordered_map<uint64_t, AllocationMetadata>();
    g_function_alloc_stack = new std::vector<std::vector<uint64_t>>();
    g_unused_timers = new std::vector<TimingRecord>();
    g_first_use_timers = new std::vector<TimingRecord>();
    g_total_allocations = new std::atomic<uint64_t>(0);
    g_read_only_count = new std::atomic<uint64_t>(0);
    g_write_only_count = new std::atomic<uint64_t>(0);
    g_read_write_count = new std::atomic<uint64_t>(0);
    g_unused_count = new std::atomic<uint64_t>(0);
    initialized = true;
  }
}

// Extract allocation ID from fake pointer
// We use bits 48-63 (top 16 bits) for the allocation ID
static inline uint64_t extract_alloc_id(void *fake_ptr) {
  uintptr_t ptr_val = reinterpret_cast<uintptr_t>(fake_ptr);
  return (ptr_val >> 48) & 0xFFFF;
}

// Extract offset from fake pointer
// We use bits 0-47 (lower 48 bits) for offsets
static inline uint64_t extract_offset(void *fake_ptr) {
  uintptr_t ptr_val = reinterpret_cast<uintptr_t>(fake_ptr);
  return ptr_val & 0x0000FFFFFFFFFFFF;
}

// Create a fake pointer from an allocation ID
static inline void *create_fake_ptr(int32_t alloc_id) {
  // Use absolute value and mask to 16 bits, place in top bits (48-63)
  uint64_t abs_id = static_cast<uint64_t>(alloc_id < 0 ? -alloc_id : alloc_id);
  uint64_t fake_val = ((abs_id & 0xFFFF) << 48);
  return reinterpret_cast<void *>(fake_val);
}

// Lookup real pointer from fake pointer and apply offset
static inline void *lookup_real_ptr(void *fake_ptr) {
  if (!g_allocation_table)
    return fake_ptr;

  uint64_t alloc_id = extract_alloc_id(fake_ptr);
  uint64_t offset = extract_offset(fake_ptr);

  auto it = g_allocation_table->find(alloc_id);
  if (it != g_allocation_table->end()) {
    // Apply offset to real pointer
    uintptr_t real_addr = reinterpret_cast<uintptr_t>(it->second.real_pointer);
    return reinterpret_cast<void *>(real_addr + offset);
  }
  return fake_ptr; // Fallback to fake_ptr if not found
}

// Update metadata on use
static inline void update_on_use(uint64_t alloc_id, bool is_read,
                                 bool is_write) {
  if (!g_allocation_table || !g_first_use_timers)
    return;

  auto it = g_allocation_table->find(alloc_id);
  if (it == g_allocation_table->end()) {
    return;
  }

  AllocationMetadata &meta = it->second;
  TimePoint now = Clock::now();

  if (!meta.has_been_used) {
    meta.first_use_time = now;
    meta.has_been_used = true;

    // Record time from allocation to first use
    Duration first_use_duration = std::chrono::duration_cast<Duration>(
        meta.first_use_time - meta.allocation_time);
    g_first_use_timers->push_back({meta.alloc_id, first_use_duration});
  }

  if (is_read) {
    meta.was_read = true;
  }
  if (is_write) {
    meta.was_written = true;
  }

  meta.last_use_time = now;
}

extern "C" {

void __pointer_tracking_pre_module() { init_globals(); }

// Called before a function starts
void __pointer_tracking_pre_function(char *name, int32_t id) {
  if (!g_function_alloc_stack)
    return;

  // Push a new empty vector for this function's allocations
  g_function_alloc_stack->push_back(std::vector<uint64_t>());
}

// Called after an alloca instruction
void *__pointer_tracking_post_alloca(void *address, int64_t size, int32_t id) {

  uint64_t abs_id = static_cast<uint64_t>(id < 0 ? -id : id);

  AllocationMetadata meta;
  meta.real_pointer = address;
  meta.size = static_cast<uint64_t>(size);
  meta.was_read = false;
  meta.was_written = false;
  meta.allocation_time = Clock::now();
  meta.has_been_used = false;
  meta.alloc_id = id;
  meta.is_global = false;
  meta.name[0] = '\0';

  (*g_allocation_table)[abs_id] = meta;
  g_total_allocations->fetch_add(1, std::memory_order_relaxed);

  // Add this allocation to the current function's scope
  if (g_function_alloc_stack && !g_function_alloc_stack->empty()) {
    g_function_alloc_stack->back().push_back(abs_id);
  }

  return create_fake_ptr(id);
}

// Called after a global variable initialization
void *__pointer_tracking_pre_global(void *address, int64_t size, char *name,
                                    int32_t id) {

  uint64_t abs_id = static_cast<uint64_t>(id < 0 ? -id : id);

  AllocationMetadata meta;
  meta.real_pointer = address;
  meta.size = static_cast<uint64_t>(size);
  meta.was_read = false;
  meta.was_written = false;
  meta.allocation_time = Clock::now();
  meta.has_been_used = false;
  meta.alloc_id = id;
  meta.is_global = true;

  if (name) {
    strncpy(meta.name, name, std::min(strlen(name), sizeof(meta.name) - 1));
    meta.name[sizeof(meta.name) - 1] = '\0';
  } else {
    meta.name[0] = '\0';
  }

  (*g_allocation_table)[abs_id] = meta;
  g_total_allocations->fetch_add(1, std::memory_order_relaxed);

  return create_fake_ptr(id);
}

// Called before a load instruction
void *__pointer_tracking_pre_load(void *pointer, int32_t id) {
  uint64_t alloc_id = extract_alloc_id(pointer);
  update_on_use(alloc_id, true, false);
  return lookup_real_ptr(pointer);
}

// Called before a store instruction
void *__pointer_tracking_pre_store(void *pointer, int32_t id) {
  uint64_t alloc_id = extract_alloc_id(pointer);
  update_on_use(alloc_id, false, true);
  return lookup_real_ptr(pointer);
}

// Called at function exit to deallocate stack allocations
void __pointer_tracking_post_function(char *name, int32_t id) {
  if (!g_allocation_table || !g_unused_timers || !g_function_alloc_stack)
    return;

  // Check if we have any function scopes
  if (g_function_alloc_stack->empty())
    return;

  TimePoint now = Clock::now();

  // Get the allocation IDs for this function scope
  std::vector<uint64_t> &alloc_ids = g_function_alloc_stack->back();

  // Process and remove only allocations from this function
  for (uint64_t alloc_id : alloc_ids) {
    auto it = g_allocation_table->find(alloc_id);
    if (it == g_allocation_table->end())
      continue;

    AllocationMetadata &meta = it->second;

    // Accumulate statistics before removing
    if (!meta.has_been_used) {
      g_unused_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_read && meta.was_written) {
      g_read_write_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_read) {
      g_read_only_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_written) {
      g_write_only_count->fetch_add(1, std::memory_order_relaxed);
    }

    // If it was used, record the time from last use to deallocation
    if (meta.has_been_used) {
      Duration unused_duration =
          std::chrono::duration_cast<Duration>(now - meta.last_use_time);
      g_unused_timers->push_back({meta.alloc_id, unused_duration});
    }

    // Remove this allocation from the table
    g_allocation_table->erase(alloc_id);
  }

  // Pop this function's allocation set from the stack
  g_function_alloc_stack->pop_back();
}

// Print statistics at program end
__attribute__((destructor(1000))) void __pointer_tracking_finalize() {
  if (!g_allocation_table || !g_total_allocations)
    return;

  // Add all remaining allocations (globals and any remaining stack) to
  // statistics
  for (const auto &entry : *g_allocation_table) {
    const AllocationMetadata &meta = entry.second;

    if (!meta.has_been_used) {
      g_unused_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_read && meta.was_written) {
      g_read_write_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_read) {
      g_read_only_count->fetch_add(1, std::memory_order_relaxed);
    } else if (meta.was_written) {
      g_write_only_count->fetch_add(1, std::memory_order_relaxed);
    }
  }

  uint64_t total_allocations =
      g_total_allocations->load(std::memory_order_relaxed);
  uint64_t read_only_count = g_read_only_count->load(std::memory_order_relaxed);
  uint64_t write_only_count =
      g_write_only_count->load(std::memory_order_relaxed);
  uint64_t read_write_count =
      g_read_write_count->load(std::memory_order_relaxed);
  uint64_t unused_count = g_unused_count->load(std::memory_order_relaxed);

  std::printf("\n");
  std::printf("=================================================\n");
  std::printf("        Pointer Tracking Statistics\n");
  std::printf("=================================================\n");
  std::printf("Total allocations tracked: %llu\n",
              static_cast<unsigned long long>(total_allocations));
  std::printf("\n");
  std::printf("Usage Patterns:\n");
  std::printf("  Read-only:           %20llu\n",
              static_cast<unsigned long long>(read_only_count));
  std::printf("  Write-only:          %20llu\n",
              static_cast<unsigned long long>(write_only_count));
  std::printf("  Read and Write:      %20llu\n",
              static_cast<unsigned long long>(read_write_count));
  std::printf("  Unused:              %20llu\n",
              static_cast<unsigned long long>(unused_count));
  std::printf("\n");

  // Sort and print top 5 longest unused times (last use to deallocation)
  if (g_unused_timers && !g_unused_timers->empty()) {
    std::sort(g_unused_timers->begin(), g_unused_timers->end(),
              [](const TimingRecord &a, const TimingRecord &b) {
                return a.duration > b.duration;
              });

    std::printf("Top 5 Longest Unused Times (last use to deallocation):\n");
    size_t unused_to_print = std::min(size_t(5), g_unused_timers->size());
    for (size_t i = 0; i < unused_to_print; i++) {
      const TimingRecord &rec = (*g_unused_timers)[i];
      std::printf("  Allocation ID %6d: %12lld ns\n", rec.alloc_id,
                  static_cast<long long>(rec.duration.count()));
    }
  } else {
    std::printf("Top 5 Longest Unused Times (last use to deallocation):\n");
    std::printf("  (none)\n");
  }
  std::printf("\n");

  // Sort and print top 5 longest first use times (allocation to first use)
  if (g_first_use_timers && !g_first_use_timers->empty()) {
    std::sort(g_first_use_timers->begin(), g_first_use_timers->end(),
              [](const TimingRecord &a, const TimingRecord &b) {
                return a.duration > b.duration;
              });

    std::printf("Top 5 Longest First Use Times (allocation to first use):\n");
    size_t first_use_to_print = std::min(size_t(5), g_first_use_timers->size());
    for (size_t i = 0; i < first_use_to_print; i++) {
      const TimingRecord &rec = (*g_first_use_timers)[i];
      std::printf("  Allocation ID %6d: %12lld ns\n", rec.alloc_id,
                  static_cast<long long>(rec.duration.count()));
    }
  } else {
    std::printf("Top 5 Longest First Use Times (allocation to first use):\n");
    std::printf("  (none)\n");
  }
  std::printf("=================================================\n");
}

} // extern "C"
