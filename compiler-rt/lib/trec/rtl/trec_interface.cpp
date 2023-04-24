//===-- trec_interface.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include "trec_interface.h"

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_ptrauth.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "trec_rtl.h"

#define CALLERPC (StackTrace::GetPreviousInstructionPc((uptr)__builtin_return_address(0)))

using namespace __trec;
using namespace __trec_metadata;

void __trec_init() {
  cur_thread_init();
  Initialize(cur_thread());
}

void __trec_flush_memory() { 
}

