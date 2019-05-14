// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -DNO_WARN_X86_INTRINSICS -target powerpc64-gnu-linux %s -Xclang -verify -o - | FileCheck %s
// RUN: %clang -S -emit-llvm -DNO_WARN_X86_INTRINSICS -target powerpc64-gnu-linux %s -Xclang -verify -x c++ -o - | FileCheck %s
// expected-no-diagnostics

// RUN: not %clang -S -emit-llvm -target powerpc64-gnu-linux %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR

#include <mmintrin.h>
// CHECK-ERROR: mmintrin.h:{{[0-9]+}}:{{[0-9]+}}: error: "Please read comment above. Use -DNO_WARN_X86_INTRINSICS to disable this error."

// CHECK: target triple = "powerpc64-
// CHECK: !llvm.module.flags =
