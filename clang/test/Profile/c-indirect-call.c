// Check the value profiling intrinsics emitted by instrumentation.

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-indirect-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck --check-prefix=NOEXT %s
// RUN: %clang_cc1 -triple s390x-ibm-linux -main-file-name c-indirect-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck --check-prefix=EXT %s

void (*foo)(void);

int main(void) {
// NOEXT:  [[REG1:%[0-9]+]] = load ptr, ptr @foo, align 8
// NOEXT-NEXT:  [[REG2:%[0-9]+]] = ptrtoint ptr [[REG1]] to i64
// NOEXT-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], ptr @__profd_main, i32 0)
// NOEXT-NEXT:  call void [[REG1]]()
// EXT:  [[REG1:%[0-9]+]] = load ptr, ptr @foo, align 8
// EXT-NEXT:  [[REG2:%[0-9]+]] = ptrtoint ptr [[REG1]] to i64
// EXT-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], ptr @__profd_main, i32 zeroext 0)
// EXT-NEXT:  call void [[REG1]]()
  foo();
  return 0;
}

// NOEXT: declare void @__llvm_profile_instrument_target(i64, ptr, i32)
// EXT: declare void @__llvm_profile_instrument_target(i64, ptr, i32 zeroext)
