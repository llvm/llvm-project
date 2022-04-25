// Check the value profiling intrinsics emitted by instrumentation.

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-avoid-direct-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck %s

void foo();

int main(void) {
// CHECK-NOT: call void @__llvm_profile_instrument_target
  foo(21);
  return 0;
}
