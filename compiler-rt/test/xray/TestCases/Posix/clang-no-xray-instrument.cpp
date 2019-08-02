// Test that we cannot actually find XRay instrumentation when we build with
// -fno-xray-instrument but have code that's marked as 'xray_always_instrument'.
//
// RUN: %clangxx -fno-xray-instrument -c %s -o %t.o
// RUN: not %llvm_xray extract -symbolize %t.o 2>&1 | FileCheck %s
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

// CHECK: llvm-xray: Cannot extract instrumentation map
// CHECK-NOT: {{.*always_instrumented.*}}
[[clang::xray_always_instrument]] int always_instrumented() { return 42; }
