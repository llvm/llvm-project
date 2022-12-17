// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Use of va_arg with a vector type exposes the vector ABI.

#include <stdarg.h>

static int bar(va_list vl) {
  return va_arg(vl, vector int)[0];
}

int foo(va_list vl) {
  return bar(vl);
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}

