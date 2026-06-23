//===-- copyprof.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the core CopyProf runtime initialization and callback
/// functions inserted by the instrumentation passes.
///
//===----------------------------------------------------------------------===//

#include "copyprof_interface_internal.h"
#include "copyprof_reporting.h"
#include "copyprof_shadow.h"
#include "copyprof_state.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

using namespace __copyprof;

namespace __copyprof {

// Whether CopyProf has been initialized.
bool copyprof_is_initialized;
// Whether CopyProf is currently initializing.
bool copyprof_init_is_running;

static void CheckUnwind() {
  UNINITIALIZED BufferedStackTrace trace;
  trace.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
               /*context=*/nullptr, common_flags()->fast_unwind_on_check);
  trace.Print();
}

static void Initialize() {
  if (LIKELY(copyprof_is_initialized))
    return;
  CHECK(!copyprof_init_is_running &&
        "BUG: CopyProf Initialize() must not call itself.");
  copyprof_init_is_running = true;
  CacheBinaryName();
  SetCheckUnwindCallback(&CheckUnwind);
  InitializePlatformEarly();
  SetCommonFlagsDefaults();
  InitializeCommonFlags();
  InitializeShadowMemory();
  copyprof_init_is_running = false;
  copyprof_is_initialized = true;
}

static void MaybeUpdateSmfContext(SmfContext context) {
  // Only entering a top level special member function changes the current
  // context. The context logically remains the same until control flow leaves
  // the top level function.
  if (__copyprof_state.smf_context == SmfContext::NONE ||
      (__copyprof_state.construct_nesting_level == 0 &&
       __copyprof_state.copy_nesting_level == 0 &&
       __copyprof_state.destruct_nesting_level == 0)) {
    __copyprof_state.smf_context = context;
  }
}

// Updates shadow memory for `[addr, addr + size)` according to the current SMF
// context: no update in `DTOR` context (objects may mutate their memory during
// destruction, but that must not invalidate its classification as an
// unnecessary copy), marked as copy in `COPY` context, and marked as non-copy
// otherwise.
static void UpdateShadow(const void* addr, uptr size) {
  if (UNLIKELY(__copyprof_state.smf_context == SmfContext::DTOR))
    return;
  MarkApplicationMemory(addr, size,
                        __copyprof_state.smf_context == SmfContext::COPY);
}

static void CopyMemberFunctionEnter(const void* this_ptr, uptr obj_size) {
  MaybeUpdateSmfContext(SmfContext::COPY);
  if (__copyprof_state.copy_nesting_level++ == 0) {
    __copyprof_state.current_this_ptr = this_ptr;
  }
  UpdateShadow(this_ptr, obj_size);
}

static void CopyMemberFunctionExit(const void* this_ptr, uptr obj_size) {
  --__copyprof_state.copy_nesting_level;
  MaybeUpdateSmfContext(SmfContext::NONE);
}

}  // namespace __copyprof

void __copyprof_init() { Initialize(); }

void __copyprof_ctor_enter_callback(const void* this_ptr, uptr obj_size) {
  MaybeUpdateSmfContext(SmfContext::CTOR);
  ++__copyprof_state.construct_nesting_level;
  UpdateShadow(this_ptr, obj_size);
}

void __copyprof_ctor_exit_callback(const void* this_ptr, uptr obj_size) {
  --__copyprof_state.construct_nesting_level;
  MaybeUpdateSmfContext(SmfContext::NONE);
}

void __copyprof_copy_ctor_enter_callback(const void* this_ptr,
                                         const void* other_ptr, uptr obj_size) {
  CopyMemberFunctionEnter(this_ptr, obj_size);
}

void __copyprof_copy_ctor_exit_callback(const void* this_ptr,
                                        const void* other_ptr, uptr obj_size) {
  CopyMemberFunctionExit(this_ptr, obj_size);
}

void __copyprof_copy_assign_op_enter_callback(const void* this_ptr,
                                              const void* other_ptr,
                                              uptr obj_size) {
  CopyMemberFunctionEnter(this_ptr, obj_size);
}

void __copyprof_copy_assign_op_exit_callback(const void* this_ptr,
                                             const void* other_ptr,
                                             uptr obj_size) {
  CopyMemberFunctionExit(this_ptr, obj_size);
}

void __copyprof_dtor_enter_callback(const void* this_ptr, uptr obj_size) {
  MaybeUpdateSmfContext(SmfContext::DTOR);
  if (__copyprof_state.destruct_nesting_level++ == 0) {
    // Control flow just entered the top-level d'tor. Optimistically mark
    // `is_transitive_copy` as `true`. If any subsequent d'tor observes object
    // memory marked as not copy, then the flag will be set to `false`.
    __copyprof_state.is_transitive_copy = true;
  }
}

void __copyprof_dtor_exit_callback(const void* this_ptr, uptr obj_size) {
  --__copyprof_state.destruct_nesting_level;
  MaybeUpdateSmfContext(SmfContext::NONE);
  // If this is the top level-dtor and `this` is transitively marked as copy,
  // then an object has been found whose transitively owned memory is marked as
  // copy, so a report is logged.
  __copyprof_state.is_transitive_copy &= IsMarkedAsCopy(this_ptr, obj_size);
  if (__copyprof_state.destruct_nesting_level == 0 &&
      __copyprof_state.is_transitive_copy) {
    // TODO: Dynamic allocation tracking via malloc/new interception will be
    // added in a subsequent patch. For now, did_allocate is set to true.
    LogCopyProfReport(GET_CALLER_PC(), GET_CURRENT_FRAME(), obj_size,
                      /*did_allocate=*/true);
  }
}

void __copyprof_store_callback(const void* addr, uptr size) {
  UpdateShadow(addr, size);
}
