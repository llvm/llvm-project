// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Globally visible pointer to a vector variable. An unused pointer doesn't
// really expose the vector ABI, but as there seems to be no easy way to
// check if a pointer is dereferenced or not (when compiling C++ at least),
// this is treated conservatively.

typedef __attribute__((vector_size(16))) int v4i32;

v4i32 *VecPtr;

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
