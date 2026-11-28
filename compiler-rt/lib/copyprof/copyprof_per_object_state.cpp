//===-- copyprof_per_object_state.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements per-object state tracking utilities for CopyProf,
/// managing transitive object sizes and allocation flags.
///
//===----------------------------------------------------------------------===//

#include "copyprof_per_object_state.h"

#include "copyprof_state.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __copyprof {
namespace {

using PerObjMap = DenseMap<const void*, PerObjectState>;

// Global map for per-object size tracking.
// Protected by mutex for thread-safe access.
StaticSpinMutex g_per_obj_map_mutex;
alignas(PerObjMap) char g_per_obj_map_placeholder[sizeof(PerObjMap)];
PerObjMap* g_per_obj_map = nullptr;

}  // namespace

void InitializePerObjectTracking() {
  g_per_obj_map_mutex.Init();
  g_per_obj_map = new (g_per_obj_map_placeholder) PerObjMap();
}

void InitPerObjectState(const void* this_ptr, uptr static_size) {
  SpinMutexLock lock(&g_per_obj_map_mutex);
  PerObjectState& state = (*g_per_obj_map)[this_ptr];
  state.SetStaticSize(static_size);
  state.did_allocate = false;
}

void RecordPerObjectAllocation(const void* this_ptr, uptr num_bytes) {
  SpinMutexLock lock(&g_per_obj_map_mutex);
  PerObjectState& state = (*g_per_obj_map)[this_ptr];
  state.IncreaseTrackedSize(num_bytes);
  state.did_allocate = true;
}

uptr GetPerObjectTrackedSize(const void* this_ptr) {
  SpinMutexLock lock(&g_per_obj_map_mutex);
  const auto* it = g_per_obj_map->find(this_ptr);
  return it ? it->second.GetTrackedSize() : 0;
}

bool GetPerObjectDidAllocate(const void* this_ptr) {
  SpinMutexLock lock(&g_per_obj_map_mutex);
  const auto* it = g_per_obj_map->find(this_ptr);
  return it ? it->second.did_allocate : false;
}

void RemovePerObjectState(const void* this_ptr) {
  SpinMutexLock lock(&g_per_obj_map_mutex);
  g_per_obj_map->erase(this_ptr);
}

}  // namespace __copyprof
