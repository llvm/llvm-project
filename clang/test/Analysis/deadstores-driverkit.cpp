/* iig(DriverKit-60) generated from SomethingSomething.iig */

// The comment above is the whole point of the test.
// That's how the suppression works.
// It needs to be on the top.
// Run-lines can wait.

// RUN: %clang_analyze_cc1 -std=c++17 -w -triple x86_64-apple-driverkit19.0 \
// RUN:   -analyzer-checker=deadcode -verify %s

// expected-no-diagnostics

#include "os_object_base.h"

class OSSomething {
  kern_return_t Invoke(const IORPC);
  void foo(OSDispatchMethod supermethod) {
    kern_return_t ret;
    IORPC rpc;
    // Test the DriverKit specific suppression in the dead stores checker.
    if (supermethod) ret = supermethod((OSObject *)this, rpc); // no-warning
    else             ret = ((OSObject *)this)->Invoke(rpc); // no-warning
  }
};
