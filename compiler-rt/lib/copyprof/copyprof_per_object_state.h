//===-- copyprof_per_object_state.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares per-object state data structures and helper functions
/// for tracking dynamic memory allocations associated with object copies.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_PER_OBJECT_STATE_H
#define COPYPROF_PER_OBJECT_STATE_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

// Initializes the per-object state tracking.
// Must be called once at program startup, after flags are initialized.
void InitializePerObjectTracking();

// Initializes the per-object state for the object at `this_ptr` with the given
// static size.
void InitPerObjectState(const void* this_ptr, uptr static_size);

// Records a dynamic allocation of `num_bytes` for the object at `this_ptr`.
void RecordPerObjectAllocation(const void* this_ptr, uptr num_bytes);

// Returns the tracked size for the object at `this_ptr`.
uptr GetPerObjectTrackedSize(const void* this_ptr);

// Returns whether the object at `this_ptr` has performed any dynamic
// allocations.
bool GetPerObjectDidAllocate(const void* this_ptr);

// Removes the per-object state for the object at `this_ptr`.
void RemovePerObjectState(const void* this_ptr);

}  // namespace __copyprof

#endif  // COPYPROF_PER_OBJECT_STATE_H
