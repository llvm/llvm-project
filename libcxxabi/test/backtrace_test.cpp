//===---------------------- backtrace_test.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <stddef.h>
#include <unwind.h>

extern "C" _Unwind_Reason_Code
trace_function(struct _Unwind_Context* context, void* ntraced) {
  (*reinterpret_cast<size_t*>(ntraced))++;
  // We should never have a call stack this deep...
  assert(*reinterpret_cast<size_t*>(ntraced) < 20);
  return _URC_NO_REASON;
}

void call3_throw(size_t* ntraced) {
  try {
    _Unwind_Backtrace(trace_function, ntraced);
  } catch (...) {
    assert(false);
  }
}

void call3_nothrow(size_t* ntraced) {
  _Unwind_Backtrace(trace_function, ntraced);
}

void call2(size_t* ntraced, bool do_throw) {
  if (do_throw) {
    call3_throw(ntraced);
  } else {
    call3_nothrow(ntraced);
  }
}

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
