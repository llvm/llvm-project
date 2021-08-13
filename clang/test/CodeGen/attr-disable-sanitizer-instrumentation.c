// RUN: %clang_cc1 -debug-info-kind=limited %s -emit-llvm -o - | FileCheck %s

void t1() __attribute__((disable_sanitizer_instrumentation)) {
}
// CHECK: disable_sanitizer_instrumentation
// CHECK-NEXT: void @t1

// CHECK-NOT: disable_sanitizer_instrumentation
// CHECK: void @t2
void t2() {
}
