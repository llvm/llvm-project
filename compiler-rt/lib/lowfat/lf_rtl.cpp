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

#include "lf_allocator.h"
#include "lf_config.h"
#include "lf_interface.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"

using namespace __sanitizer;

namespace __lowfat {

// Flag to track initialization state (not static — accessed by lf_interceptors.cpp)
bool lowfat_inited = false;

// Set to true when -fsanitize-recover=lowfat is active. Controls whether
// interceptor-level OOB (memset/memcpy/memmove) warns-and-continues or aborts.
bool lowfat_recover = false;

// Maximum number of size classes across both modes.
// In POW2-only mode kNumSizeClasses=27; with a custom config it can be up to
// LOWFAT_MAX_ARRAY_SIZE. We size the static arrays at compile time using
// whichever is larger so the same translation unit works in both modes.
#ifdef LOWFAT_CUSTOM_CONFIG
  static constexpr uptr kMaxSizeClasses = LOWFAT_NUM_SIZE_CLASSES;
#else
  static constexpr uptr kMaxSizeClasses = kNumSizeClasses;
#endif

// Region table - initialized in __lf_init
RegionInfo kRegions[kMaxSizeClasses];

// Pointers to the start of each mapped region
static uptr region_bases[kMaxSizeClasses];

// Bump pointer: next fresh address to allocate from in each region
static uptr region_next_alloc[kMaxSizeClasses];

// Segregated free lists: one singly-linked list per size class
// Free blocks store a pointer to the next free block at their start
struct FreeBlock {
  FreeBlock *next;
};
static FreeBlock *free_lists[kMaxSizeClasses];

// Per-size-class spin mutexes protecting region_next_alloc and free_lists.
// Using one lock per size class allows concurrent allocation across different
// size classes, which is the common case in multi-threaded programs.
static StaticSpinMutex region_locks[kMaxSizeClasses];

static void InitializeFlags() {
  SetCommonFlagsDefaults();

  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.exitcode = 1; // Fatal OOB exits with code 1 by default
    cf.abort_on_error = false; // Use the exitcode path, not SIGABRT, so output is flushed before the process exits
    OverrideCommonFlags(cf);
  }

  // Register all common flags with a parser and read LOWFAT_OPTIONS.
  //    Allow overriding flags at runtime, e.g.: LOWFAT_OPTIONS=exitcode=42:verbosity=1 ./my_program
  FlagParser parser;
  RegisterCommonFlags(&parser);
  parser.ParseStringFromEnv("LOWFAT_OPTIONS");

  InitializeCommonFlags();
}

static void InitRegionTable() {
  for (uptr i = 0; i < kNumSizeClasses; i++) {
    uptr size = SizeClassToSize(i);
    kRegions[i].size      = size;
    kRegions[i].alignment = size;
#ifdef LOWFAT_CUSTOM_CONFIG
    // For non-POW2 sizes the mask is meaningless (base computed via magic
    // multiply); store 0 to make this explicit and catch accidental usage.
    kRegions[i].mask = kLowFatGenIsPow2[i] ? (uptr)kLowFatGenMasks[i] : 0;
#else
    kRegions[i].mask = ~(size - 1);
#endif
    free_lists[i] = nullptr;
  }
}

// Initialize memory regions using mmap
// Each region is mapped at a fixed address for the corresponding size class
static bool InitMemoryRegions() {
  for (uptr i = 0; i < kNumSizeClasses; i++) {
    uptr region_start = GetRegionStart(i);
    
    // Reserve the region without committing physical memory
    // MmapFixedNoReserve maps memory but doesn't allocate physical pages until they're accessed
    bool success = MmapFixedNoReserve(region_start, kRegionSize, "lowfat_region");
    
    if (!success)
      return false;
    
    region_bases[i] = region_start;
    
    // The first allocation must be aligned to the object size relative to absolute zero.
    // This is required for the magic-number fixed point math to securely compute object bases.
    uptr size = kRegions[i].size;
    uptr offset = region_start % size;
    uptr initial_alloc = region_start;
    if (offset != 0)
      initial_alloc += (size - offset);
      
    region_next_alloc[i] = initial_alloc;
  }
  
  return true;
}

// Allocate from a LowFat region
// First checks the free list, then falls back to bump allocation.
// Thread-safe: protected by per-size-class spin mutex.
void *Allocate(uptr size) {
  if (size == 0)
    size = 1;

  uptr class_index = SizeClassIndex(size);
  if (class_index >= kNumSizeClasses)
    return nullptr;

  uptr alloc_size = SizeClassToSize(class_index);

  SpinMutexLock lock(&region_locks[class_index]);

  // 1. Try free list first
  FreeBlock *block = free_lists[class_index];
  if (block) {
    free_lists[class_index] = block->next;
    // Zero the memory (free list pointer was stored here)
    internal_memset(block, 0, alloc_size);
    return (void *)block;
  }

  // 2. Fall back to bump allocation
  uptr region_end = GetRegionStart(class_index) + kRegionSize;
  uptr addr = region_next_alloc[class_index];

  if (addr + alloc_size > region_end) {
    // Region exhausted: fall back to standard libc-style allocation.
    // The resulting pointer will not be a LowFat pointer (wide-bounds).
    return (void *)InternalAlloc(size);
  }

  region_next_alloc[class_index] = addr + alloc_size;
  return (void *)addr;
}

// Free a LowFat allocation by pushing it onto the free list.
// Thread-safe: protected by per-size-class spin mutex.
void Deallocate(void *ptr) {
  if (!ptr)
    return;

  uptr addr = (uptr)ptr;

  // Validate this is a LowFat pointer
  if (!IsLowFatPointer(addr)) {
    InternalFree(ptr);
    return;
  }

  uptr region = GetRegionIndex(addr);

  SpinMutexLock lock(&region_locks[region]);

  // Push to the head of the free list for this size class
  FreeBlock *block = (FreeBlock *)ptr;
  block->next = free_lists[region];
  free_lists[region] = block;
}

static void PrintOobHeader(const char *level, uptr ptr, uptr base, uptr bound,
                           int is_write) {
  // Compute the signed overflow: how many bytes past the end of the allocation
  // the access reached. 0 means exactly at the boundary.
  sptr overflow = (sptr)(ptr) - (sptr)(base + bound);
  const char *op = is_write ? "write" : "read";

  Printf("LOWFAT %s: out-of-bounds error detected!\n", level);
  Printf("          operation = %s\n", op);
  Printf("          pointer   = 0x%zx (heap)\n", ptr);
  Printf("          base      = 0x%zx\n", base);
  Printf("          size      = %zu\n", bound);
  const char *sign = (overflow >= 0) ? "+" : "";
  Printf("          overflow  = %s%zd\n", sign, (long)overflow);
  Printf("\n");
}

static void PrintErrorAndDie(uptr ptr, uptr base, uptr bound, int is_write) {
  PrintOobHeader("ERROR", ptr, base, bound, is_write);
  Die();
}

static void PrintWarning(uptr ptr, uptr base, uptr bound, int is_write) {
  PrintOobHeader("WARNING", ptr, base, bound, is_write);
}

}  // namespace __lowfat

// ---------------------- Interface Functions ----------------------

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_set_recover(int recover) {
  __lowfat::lowfat_recover = (recover != 0);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_init() {
  if (__lowfat::lowfat_inited)
    return;

  __lowfat::InitializeFlags();

  __lowfat::InitRegionTable();

  if (!__lowfat::InitMemoryRegions())
    Die();

  __lowfat::lowfat_inited = true;

  __lowfat::InitializeInterceptors();
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_report_oob(uptr ptr, uptr base, uptr bound, int is_write) {
  __lowfat::PrintErrorAndDie(ptr, base, bound, is_write);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_warn_oob(uptr ptr, uptr base, uptr bound, int is_write) {
  __lowfat::PrintWarning(ptr, base, bound, is_write);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_base(uptr ptr) {
  return __lowfat::GetBase(ptr);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_size(uptr ptr) {
  return __lowfat::GetSize(ptr);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_offset(uptr ptr) {
  uptr base = __lowfat::GetBase(ptr);
  if (base == 0)
    return 0;
  return ptr - base;
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_usable_size(uptr ptr) {
  uptr base = __lowfat::GetBase(ptr);
  uptr size = __lowfat::GetSize(ptr);
  if (base == 0)
    return (uptr)-1;
  return size - (ptr - base);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__lf_malloc(uptr size) {
  return __lowfat::Allocate(size);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_free(void *ptr) {
  __lowfat::Deallocate(ptr);
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