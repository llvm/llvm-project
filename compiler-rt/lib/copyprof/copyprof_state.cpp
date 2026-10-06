//===-- copyprof_state.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the per-thread and per-object state management for
/// tracking execution context within special member functions.
///
//===----------------------------------------------------------------------===//

#include "copyprof_state.h"

namespace __copyprof {

THREADLOCAL PerThreadState __copyprof_state;

}  // namespace __copyprof
