// Check the value profiling instrinsics emitted by instrumentation.

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.9 -main-file-name c-indirect-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck --check-prefix=NOEXT %s
// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-ibm-linux -main-file-name c-indirect-call.c %s -o - -emit-llvm -fprofile-instrument=clang -mllvm -enable-value-profiling | FileCheck --check-prefix=EXT %s

void (*foo)(void);

int main(void) {
// NOEXT:  [[REG1:%[0-9]+]] = load void ()*, void ()** @foo, align 8
// NOEXT-NEXT:  [[REG2:%[0-9]+]] = ptrtoint void ()* [[REG1]] to i64
// NOEXT-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], i8* bitcast ({{.*}}* @__profd_main to i8*), i32 0)
// NOEXT-NEXT:  call void [[REG1]]()
// EXT:  [[REG1:%[0-9]+]] = load void ()*, void ()** @foo, align 8
// EXT-NEXT:  [[REG2:%[0-9]+]] = ptrtoint void ()* [[REG1]] to i64
// EXT-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], i8* bitcast ({{.*}}* @__profd_main to i8*), i32 zeroext 0)
// EXT-NEXT:  call void [[REG1]]()
  foo();
  return 0;
}

// NOEXT: declare void @__llvm_profile_instrument_target(i64, i8*, i32)
// EXT: declare void @__llvm_profile_instrument_target(i64, i8*, i32 zeroext)
