//===---------------------- backtrace_test.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-exceptions

#include <assert.h>
#include <stddef.h>
#include <unwind.h>

extern "C" _Unwind_Reason_Code
trace_function(struct _Unwind_Context*, void* ntraced) {
  (*reinterpret_cast<size_t*>(ntraced))++;
  // We should never have a call stack this deep...
  assert(*reinterpret_cast<size_t*>(ntraced) < 20);
  return _URC_NO_REASON;
}

__attribute__ ((__noinline__))
void call3_throw(size_t* ntraced) {
  try {
    _Unwind_Backtrace(trace_function, ntraced);
  } catch (...) {
    assert(false);
  }
}

__attribute__ ((__noinline__, __disable_tail_calls__))
void call3_nothrow(size_t* ntraced) {
  _Unwind_Backtrace(trace_function, ntraced);
}

__attribute__ ((__noinline__, __disable_tail_calls__))
void call2(size_t* ntraced, bool do_throw) {
  if (do_throw) {
    call3_throw(ntraced);
  } else {
    call3_nothrow(ntraced);
  }
}

__attribute__ ((__noinline__, __disable_tail_calls__))
void call1(size_t* ntraced, bool do_throw) {
  call2(ntraced, do_throw);
}

int main() {
  size_t throw_ntraced = 0;
  size_t nothrow_ntraced = 0;

  call1(&nothrow_ntraced, false);

  try {
    call1(&throw_ntraced, true);
  } catch (...) {
    assert(false);
  }

  // Different platforms (and different runtimes) will unwind a different number
  // of times, so we can't make any better assumptions than this.
  assert(nothrow_ntraced > 1);
  assert(throw_ntraced == nothrow_ntraced); // Make sure we unwind through catch
  return 0;
}
