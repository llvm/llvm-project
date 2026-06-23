//===-- copyprof_state.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the per-thread and per-object state structures used to
/// track special member function nesting and dynamic memory allocations.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_STATE_H
#define COPYPROF_STATE_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

// Determines the special member function (SMF) execution context of a thread,
// dictating how stores and allocations affect shadow memory.
// The first time control flow reaches the entry point of a special member
// function, the current SMF context is changed accordingly.
// A copy c'tor or copy assignment operator sets the context to `COPY`, a c'tor
// to `CTOR`, and a d'tor to `DTOR`.
// In the `COPY` context, shadow memory is marked as copy.
// In the `CTOR` context, shadow memory is marked as non-copy.
// In the `DTOR` context, no shadow memory is updated at all (to
// avoid false negatives during destruction). Once control flow leaves (any
// nested) special member functions, the context is set to `NONE`. In this
// state, any stores mark shadow memory as non-copy.
enum class SmfContext : u8 {
  NONE,
  CTOR,
  COPY,
  DTOR,
};

// CopyProf uses per-thread state to figure out whether control flow is
// currently inside a special member function, and adapts updating of shadow
// memory accordingly (see SmfContext). Since special member functions can
// nest arbitrarily, this state needs to be kept across function calls, so this
// state is stored in TLS. The nesting level counters are used to determine
// whether a top-level (i.e. the first in the call stack of special member
// functions) special member function has been reached.
struct PerThreadState {
  u32 construct_nesting_level = 0;
  u32 copy_nesting_level = 0;
  u32 destruct_nesting_level = 0;
  SmfContext smf_context = SmfContext::NONE;
  // Whether all transitively reachable d'tors from the top-level d'tor have
  // observed copies.
  bool is_transitive_copy = false;
  // When control flow enters a special member function, this is set to the
  // `this` pointer of the current object. This is used to look up the
  // per-object state outside of special member functions (e.g. when allocating
  // memory).
  const void* current_this_ptr = nullptr;
};

// The runtime is always linked into the main executable, so the state can be
// reached with the initial-exec model instead of paying for a __tls_get_addr
// call on every access.
__attribute__((tls_model("initial-exec")))
extern THREADLOCAL PerThreadState __copyprof_state;

}  // namespace __copyprof

#endif  // COPYPROF_STATE_H
