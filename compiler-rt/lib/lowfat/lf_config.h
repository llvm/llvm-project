//===-- lf_config.h - LowFat Memory Layout Configuration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowFat memory layout configuration.
//
// LowFat pointers encode allocation bounds directly in the pointer value:
// - Memory is divided into regions, each for a specific size class
// - Within each region, allocations are aligned to their size class
// - Given a pointer, the base can be computed by masking off low bits (POW2)
//   or via fixed-point magic-number math (non-POW2)
// - The size can be looked up from a table using the region index
//
// Default Memory Layout (POW2-only mode, kRegionSizeLog=32):
//   Region 0: [0x10_0000_0000, 0x20_0000_0000) - 16-byte allocations
//   Region 1: [0x20_0000_0000, 0x30_0000_0000) - 32-byte allocations
//   Region 2: [0x30_0000_0000, 0x40_0000_0000) - 64-byte allocations
//   ...
//   Region N: [0xN0_0000_0000, ...)            - 2^(N+4)-byte allocations
//
// Custom Config Mode (LOWFAT_CUSTOM_CONFIG, kRegionSizeLog=35):
//   Non-POW2 sizes (e.g. 48, 80, 96 bytes) are also supported.
//   kRegionSizeLog increases to 35 (32 GB per region) to preserve precision
//   of the magic-number arithmetic across the full region.
//   The key helpers (SizeClassIndex, SizeClassToSize) switch to table lookups.
//   All other logic (GetBase, GetSize, CheckBounds, etc.) is unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef LF_CONFIG_H
#define LF_CONFIG_H

#include "sanitizer_common/sanitizer_internal_defs.h"

#ifdef LOWFAT_CUSTOM_CONFIG
#include "lf_config_generated.h"
#endif

namespace __lowfat {

using namespace __sanitizer;

//===----------------------------------------------------------------------===//
// Size Class Configuration
//===----------------------------------------------------------------------===//

// Minimum allocation size (must be power of 2)
constexpr uptr kMinSizeLog = 4;  // 16 bytes
constexpr uptr kMinSize = 1ULL << kMinSizeLog;

#ifdef LOWFAT_CUSTOM_CONFIG

constexpr uptr kNumSizeClasses = LOWFAT_NUM_SIZE_CLASSES;
constexpr uptr kMaxSize = LOWFAT_MAX_SIZE;

// SizeClassIndex: table lookup (binary search on kLowFatGenSizes[])
// Replaces the POW2-only __builtin_clzll math.
inline uptr SizeClassIndex(uptr size) {
  return (uptr)lowfat_size_to_class((uint64_t)size);
}

// SizeClassToSize: direct table lookup — works for both POW2 and non-POW2.
inline uptr SizeClassToSize(uptr class_index) {
  if (class_index >= kNumSizeClasses)
    return 0;
  return (uptr)kLowFatGenSizes[class_index];
}

#else

// Maximum allocation size (must be power of 2)
constexpr uptr kMaxSizeLog = 30;  // 1 GB
constexpr uptr kMaxSize = 1ULL << kMaxSizeLog;

// Number of size classes (one per power of 2)
constexpr uptr kNumSizeClasses = kMaxSizeLog - kMinSizeLog + 1;

// Size class index for a given size (rounded up to next power of 2)
// Returns 0 for sizes <= 16, 1 for sizes 17-32, etc.
inline uptr SizeClassIndex(uptr size) {
  if (size <= kMinSize)
    return 0;
  // Count leading zeros to find the highest set bit
  uptr log2 = (sizeof(uptr) * 8 - 1) - __builtin_clzll(size);
  // Round up if not exact power of 2
  if (size > (1ULL << log2))
    log2++;
  return log2 - kMinSizeLog;
}

// Get the allocation size for a size class
inline uptr SizeClassToSize(uptr class_index) {
  return 1ULL << (class_index + kMinSizeLog);
}

#endif // LOWFAT_CUSTOM_CONFIG

//===----------------------------------------------------------------------===//
// Memory Region Configuration
//===----------------------------------------------------------------------===//

#ifdef LOWFAT_CUSTOM_CONFIG
constexpr uptr kRegionSizeLog = LOWFAT_REGION_SIZE_LOG; // 35
#else
// Each region is 4GB (32 bits of address space per region)
constexpr uptr kRegionSizeLog = 32;
#endif

constexpr uptr kRegionSize = 1ULL << kRegionSizeLog;

// Base address where LowFat regions start
// We use the upper portion of the address space
// On 64-bit systems: 0x100000000000 (17.6 TB mark)
constexpr uptr kRegionBase = 0x100000000000ULL;

// Get the region number from a pointer
inline uptr GetRegionIndex(uptr ptr) {
  if (ptr < kRegionBase)
    return (uptr)-1;  // Not a LowFat pointer
  return (ptr - kRegionBase) >> kRegionSizeLog;
}

// Get the start address of a region
inline uptr GetRegionStart(uptr region_index) {
  return kRegionBase + (region_index << kRegionSizeLog);
}

// Check if a pointer is within LowFat managed memory
inline bool IsLowFatPointer(uptr ptr) {
  uptr region = GetRegionIndex(ptr);
  return region < kNumSizeClasses;
}

//===----------------------------------------------------------------------===//
// Bounds Computation
//===----------------------------------------------------------------------===//

// Get the allocation size from a LowFat pointer
inline uptr GetSize(uptr ptr) {
  uptr region = GetRegionIndex(ptr);
  if (region >= kNumSizeClasses)
    return (uptr)-1;  // Wide-bounds for non-LowFat pointers
  return SizeClassToSize(region);
}

#ifdef LOWFAT_CUSTOM_CONFIG

// GetBase override for non-POW2: use magic-number multiplication instead
// of the bitwise-AND fast path when the size class is not a power of two.
//
// For POW2 sizes:    base = ptr & ~(size - 1)           [fast path]
// For non-POW2:      base = ((u128)ptr * magic >> 64) * size  [magic path]
inline uptr GetBase(uptr ptr) {
  uptr region = GetRegionIndex(ptr);
  if (region >= kNumSizeClasses)
    return 0;
  if (kLowFatGenIsPow2[region]) {
    // Fast path: bitwise AND
    return ptr & (uptr)kLowFatGenMasks[region];
  } else {
    // Magic-number fixed-point path
    typedef unsigned __int128 u128;
    u128 mul  = (u128)ptr * (u128)kLowFatGenMagics[region];
    uptr idx  = (uptr)(mul >> 64);
    return idx * (uptr)kLowFatGenSizes[region];
  }
}

// CheckBounds override: uses the custom GetBase above.
inline bool CheckBounds(uptr ptr, uptr access_size) {
  uptr region = GetRegionIndex(ptr);
  if (region >= kNumSizeClasses)
    return true;  // Not a LowFat pointer — assume valid
  uptr alloc_size = SizeClassToSize(region);
  uptr base       = GetBase(ptr);
  uptr end        = base + alloc_size;
  return (ptr + access_size) <= end;
}

#else

// Get the base address of an allocation from a LowFat pointer
// This uses the key LowFat insight: allocations are aligned to their size
inline uptr GetBase(uptr ptr) {
  uptr region = GetRegionIndex(ptr);
  if (region >= kNumSizeClasses)
    return 0;  // Not a valid LowFat pointer
  
  uptr size = SizeClassToSize(region);
  uptr mask = ~(size - 1);  // Mask off low bits
  return ptr & mask;
}

// Check if ptr..ptr+access_size is within bounds
inline bool CheckBounds(uptr ptr, uptr access_size) {
  uptr region = GetRegionIndex(ptr);
  if (region >= kNumSizeClasses)
    return true;  // Not a LowFat pointer, assume valid (or could error)
  
  uptr alloc_size = SizeClassToSize(region);
  uptr base = ptr & ~(alloc_size - 1);
  uptr end = base + alloc_size;
  
  return (ptr + access_size) <= end;
}

#endif // LOWFAT_CUSTOM_CONFIG

//===----------------------------------------------------------------------===//
// Region Table (for lookup by region index)
//===----------------------------------------------------------------------===//

struct RegionInfo {
  uptr size;           // Allocation size for this region
  uptr alignment;      // Alignment (same as size for LowFat)
  uptr mask;           // Mask to get base address: ptr & mask
};

// This table is indexed by region number
// Initialized in lf_rtl.cpp
extern RegionInfo kRegions[kNumSizeClasses];

}  // namespace __lowfat

#endif  // LF_CONFIG_H
