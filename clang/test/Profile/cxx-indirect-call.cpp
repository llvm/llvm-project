// Check the value profiling intrinsics emitted by instrumentation.

// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling -fexceptions -fcxx-exceptions -triple x86_64-apple-macosx10.9 | FileCheck %s

void (*foo) (void);

int main(int argc, const char *argv[]) {
// CHECK:  [[REG1:%[0-9]+]] = load ptr, ptr @foo
// CHECK-NEXT:  [[REG2:%[0-9]+]] = ptrtoint ptr [[REG1]] to i64
// CHECK-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], ptr @__profd_main, i32 0)
// CHECK-NEXT:  invoke void [[REG1]]()
  try {
    foo();
  } catch (int) {}
  return 0;
}

// CHECK: declare void @__llvm_profile_instrument_target(i64, ptr, i32)



