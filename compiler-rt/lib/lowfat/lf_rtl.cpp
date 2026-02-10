//===-- lf_rtl.cpp - LowFat Sanitizer Runtime Library ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the main file of the LowFat Sanitizer runtime library.
//
// LowFat pointers encode allocation bounds information directly in the pointer
// value through careful memory layout. This allows O(1) bounds checking without
// maintaining separate metadata.
//
//===----------------------------------------------------------------------===//

#include "lf_interface.h"
#include "lf_config.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __sanitizer;

namespace __lowfat {

// Flag to track initialization state
static bool lowfat_inited = false;

// Region table - initialized in __lf_init
// TODO: not actually needed, to use for convenience
RegionInfo kRegions[kNumSizeClasses];

// Pointers to the start of each mapped region
static uptr region_bases[kNumSizeClasses];

// Free list heads for each region (simple bump allocator for now)
static uptr region_next_alloc[kNumSizeClasses];

static void InitRegionTable() {
  for (uptr i = 0; i < kNumSizeClasses; i++) {
    uptr size = SizeClassToSize(i);
    kRegions[i].size = size;
    kRegions[i].alignment = size;
    kRegions[i].mask = ~(size - 1);
  }
}

// Initialize memory regions using mmap
// Each region is mapped at a fixed address for the corresponding size class
static bool InitMemoryRegions() {
  for (uptr i = 0; i < kNumSizeClasses; i++) {
    uptr region_start = GetRegionStart(i);
    
    // Reserve the region without committing physical memory
    // MmapFixedNoReserve maps memory but doesn't allocate physical pages
    // until they're accessed (lazy allocation)
    bool success = MmapFixedNoReserve(region_start, kRegionSize, "lowfat_region");
    
    if (!success) {
      Printf("LowFat: Failed to map region %zu at 0x%zx\n", i, region_start);
      return false;
    }
    
    region_bases[i] = region_start;
    region_next_alloc[i] = region_start;
  }
  
  Printf("LowFat: Mapped %zu regions starting at 0x%zx\n", 
         kNumSizeClasses, kRegionBase);
  return true;
}

static void PrintErrorAndDie(uptr ptr, uptr base, uptr bound) {
  Printf("=================================================================\n");
  Printf("==ERROR: LowFat: out-of-bounds access detected\n");
  Printf("  pointer: 0x%zx\n", ptr);
  Printf("  base:    0x%zx\n", base);
  Printf("  bound:   %zu bytes\n", bound);
  Printf("=================================================================\n");

  // Print stack trace
  BufferedStackTrace stack;
  stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), nullptr,
               common_flags()->fast_unwind_on_fatal);
  stack.Print();

  Die();
}

}  // namespace __lowfat

namespace __sanitizer {
void BufferedStackTrace::UnwindImpl(uptr pc, uptr bp, void *context,
                                    bool request_fast, u32 max_depth) {
  uptr top = 0;
  uptr bottom = 0;
  GetThreadStackTopAndBottom(false, &top, &bottom);
  bool fast = StackTrace::WillUseFastUnwind(request_fast);
  Unwind(max_depth, pc, bp, context, top, bottom, fast);
}
}  // namespace __sanitizer

// ---------------------- Interface Functions ----------------------

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_init() {
  if (__lowfat::lowfat_inited)
    return;

  Printf("LowFat Sanitizer: initializing runtime\n");

  __lowfat::InitRegionTable();
  
  if (!__lowfat::InitMemoryRegions()) {
    Printf("LowFat Sanitizer: failed to initialize memory regions\n");
    Die();
  }

  Printf("LowFat Sanitizer: initialized runtime\n");

  __lowfat::lowfat_inited = true;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_check_bounds(uptr ptr, uptr size) {
  if (!__lowfat::IsLowFatPointer(ptr)) { // Not a LowFat-managed pointer, skip check
    return;
  }

  if (!__lowfat::CheckBounds(ptr, size)) {
    uptr base = __lowfat::GetBase(ptr);
    uptr alloc_size = __lowfat::GetSize(ptr);
    __lowfat::PrintErrorAndDie(ptr, base, alloc_size);
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_report_oob(uptr ptr, uptr base, uptr bound) {
  __lowfat::PrintErrorAndDie(ptr, base, bound);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_base(uptr ptr) {
  return __lowfat::GetBase(ptr);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_size(uptr ptr) {
  return __lowfat::GetSize(ptr);
}

}  // extern "C"

#if SANITIZER_CAN_USE_PREINIT_ARRAY
// ELF platforms: use .preinit_array for earliest possible initialization
__attribute__((section(".preinit_array"), used)) static auto preinit =
    __lf_init;
#else
// macOS/other platforms: use constructor attribute
__attribute__((constructor)) static void lowfat_constructor() {
  __lf_init();
}
#endif