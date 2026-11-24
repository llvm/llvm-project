//===-- copyprof_shadow.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the shadow memory interface and template helpers for
/// mapping application memory to CopyProf shadow memory.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_SHADOW_H
#define COPYPROF_SHADOW_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

// FIXME: copyprof uses a 1:8 mapping but this may lead to data races on shadow
// memory for concurrent stores within the same 8 byte region. Consider using a
// 1:1 mapping or reducing granularity (e.g. atomically store whole bytes for
// each update to an 8 byte region).
constexpr uptr kShadowScale = 3;

// Helper for mapping application addresses to shadow memory.
struct ShadowMemory {
  static uptr MemToShadowSize(uptr size) { return size >> kShadowScale; }
  static ShadowMemory Create(const char* name) {
    uptr max_user_va = GetMaxUserVirtualAddress();
    uptr shadow_size_bytes =
        RoundUpTo(MemToShadowSize(max_user_va), GetMmapGranularity());
    uptr mapped = MapDynamicShadow(shadow_size_bytes, kShadowScale,
                                   /*min_shadow_base_alignment=*/0, max_user_va,
                                   GetMmapGranularity());
    ReserveShadowMemoryRange(mapped, mapped + shadow_size_bytes - 1, name,
                             /*madvise_shadow=*/true);
    return ShadowMemory(mapped);
  }
  ShadowMemory() = default;
  uptr MemToShadow(uptr p) const { return (p >> kShadowScale) + shadow_base_; }

 private:
  explicit ShadowMemory(uptr shadow_base) : shadow_base_(shadow_base) {}
  uptr shadow_base_ = 0;
};

// Must be called exactly once at program startup.
void InitializeShadowMemory();

// Given an application memory block starting at `app_addr` of size `num_bytes`,
// marks the corresponding shadow memory as a copy or non-copy.
void MarkApplicationMemory(const void* app_addr, uptr num_bytes, bool is_copy);

// Whether the application memory block starting at `app_addr` of size
// `num_bytes` is marked as a copy in shadow memory.
bool IsMarkedAsCopy(const void* app_addr, uptr num_bytes);

}  // namespace __copyprof

#endif  // COPYPROF_SHADOW_H
