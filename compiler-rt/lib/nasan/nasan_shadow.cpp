//===-- nasan_shadow.cpp - NoAliasSanitizer Shadow Memory ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shadow memory implementation for NoAliasSanitizer.
//
//===----------------------------------------------------------------------===//

#include "nasan_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_posix.h"

namespace __nasan {

using namespace __sanitizer;

// Shadow memory constants
// For pointer provenance tracking, we need each pointer-sized region (8 bytes)
// to have its own PointerShadow (8 bytes). We use a 1:1 mapping.
// Shadow address = kShadowOffset + (app_addr / 8) * sizeof(PointerShadow)
// Since sizeof(PointerShadow) == 8, this simplifies to:
// Shadow address = kShadowOffset + (app_addr & ~7ULL)
// But we need to avoid overlap, so we scale: kShadowOffset + app_addr
// This works because we map a large enough shadow region.

// Platform-specific shadow configuration
// On macOS/Apple, virtual address space is more limited, so we use a smaller region.
#if SANITIZER_APPLE
  // macOS/iOS has stricter VM limits; use a smaller shadow in a safe region
  #if defined(__x86_64__)
    static constexpr uptr kShadowOffset = 0x200000000000ULL;
  #else  // ARM64
    static constexpr uptr kShadowOffset = 0x100000000000ULL;
  #endif
  static constexpr uptr kShadowSize = (1ULL << 34);  // 16 GB
  static constexpr uptr kShadowScale = 3;  // 1 shadow byte per 8 app bytes
#elif defined(__x86_64__) && defined(__linux__)
  static constexpr uptr kShadowOffset = 0x100000000000ULL;
  static constexpr uptr kShadowSize = (1ULL << 44);  // 16 TB
  static constexpr uptr kShadowScale = 3;
#elif defined(__aarch64__)
  static constexpr uptr kShadowOffset = 0x100000000000ULL;
  static constexpr uptr kShadowSize = (1ULL << 44);  // 16 TB
  static constexpr uptr kShadowScale = 3;
#else
  #error "Unsupported platform for NASan"
#endif

// Track if shadow memory was successfully initialized
static bool g_shadow_initialized = false;
static uptr g_shadow_base = 0;
static uptr g_shadow_end = 0;

// Compute shadow address for a given application address
// Each 8-byte aligned application address maps to its own PointerShadow entry
static inline PointerShadow *get_shadow_impl(void *ptr) {
  uptr addr = reinterpret_cast<uptr>(ptr);
  // Align down to 8 bytes, then compute shadow offset
  // shadow_addr = kShadowOffset + (aligned_addr >> kShadowScale) * sizeof(PointerShadow)
  // Since sizeof(PointerShadow) == 8 == (1 << 3) and kShadowScale == 3, this simplifies to:
  // shadow_addr = kShadowOffset + (aligned_addr & ~7ULL)
  // But this causes overlap! We need: kShadowOffset + ((addr >> 3) << 3) = kShadowOffset + (addr & ~7)
  // Actually the correct formula for 1:1 mapping per 8-byte region:
  uptr aligned_addr = addr & ~7ULL;
  uptr shadow_addr = kShadowOffset + (aligned_addr >> kShadowScale) * sizeof(PointerShadow);
  return reinterpret_cast<PointerShadow *>(shadow_addr);
}

PointerShadow *get_shadow(void *ptr) {
  return get_shadow_impl(ptr);
}

void init_shadow_memory() {
  // Use sanitizer_common's MmapFixedNoReserve which handles platform differences
  uptr shadow_start = kShadowOffset;
  uptr shadow_size = kShadowSize;

  // Try to reserve the shadow memory region using sanitizer_common utilities
  if (!MmapFixedNoReserve(shadow_start, shadow_size, "nasan shadow")) {
    // If that fails, try a smaller region
    shadow_size = (1ULL << 32);  // 4 GB fallback
    if (!MmapFixedNoReserve(shadow_start, shadow_size, "nasan shadow")) {
      Printf("NASan: FATAL: Failed to map shadow memory at 0x%zx (size %zu)\n",
             shadow_start, shadow_size);
      Printf("NASan: This may be due to address space limitations.\n");
      Printf("NASan: Try running with ASLR disabled or as root.\n");
      Die();
    }
    Printf("NASan: Warning - using smaller shadow region (%zu bytes)\n", shadow_size);
  }

  g_shadow_base = shadow_start;
  g_shadow_end = shadow_start + shadow_size;
  g_shadow_initialized = true;
  atomic_store(&g_stats.shadow_bytes, shadow_size, memory_order_relaxed);
}

bool is_shadow_initialized() {
  return g_shadow_initialized;
}

} // namespace __nasan

using namespace __nasan;

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_set_pointer_provenance(void *ptr, ProvenanceID prov) {
  if (!ptr) return;
  PointerShadow *shadow = get_shadow(ptr);
  // Use atomic store for thread safety
  atomic_store(reinterpret_cast<atomic_uint64_t *>(&shadow->primary_prov),
               prov, memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
ProvenanceID __nasan_get_pointer_provenance(void *ptr) {
  if (!ptr) return 0;
  PointerShadow *shadow = get_shadow(ptr);
  // Use atomic load for thread safety
  return atomic_load(reinterpret_cast<atomic_uint64_t *>(&shadow->primary_prov),
                     memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __nasan_inherit_provenance(void *dst_ptr, void *src_ptr) {
  if (!dst_ptr) return;
  ProvenanceID prov = __nasan_get_pointer_provenance(src_ptr);
  __nasan_set_pointer_provenance(dst_ptr, prov);
}

} // extern "C"
