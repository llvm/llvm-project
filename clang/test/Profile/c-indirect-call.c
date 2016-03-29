// Check the value profiling instrinsics emitted by instrumentation.

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-indirect-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck %s

void (*foo)(void);

int main(void) {
// CHECK:  [[REG1:%[0-9]+]] = load void ()*, void ()** @foo, align 8
// CHECK-NEXT:  [[REG2:%[0-9]+]] = ptrtoint void ()* [[REG1]] to i64
// CHECK-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], i8* bitcast ({{.*}}* @__profd_main to i8*), i32 0)
// CHECK-NEXT:  call void [[REG1]]()
  foo();
  return 0;
}

// CHECK: declare void @__llvm_profile_instrument_target(i64, i8*, i32)
