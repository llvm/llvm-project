// RUN: %clang_cc1 -ffreestanding %s -triple=i686-- -target-feature +rdpru -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-- -target-feature +rdpru -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-- -target-cpu znver2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

// NOTE: This should correspond to the tests in llvm/test/CodeGen/X86/rdpru.ll

unsigned long long test_rdpru(int regid) {
  // CHECK-LABEL: test_rdpru
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.x86.rdpru(i32 %{{.*}})
  // CHECK-NEXT: ret i64 [[RESULT]]
  return __rdpru(regid);
}

unsigned long long test_mperf() {
  // CHECK-LABEL: test_mperf
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.x86.rdpru(i32 0)
  // CHECK-NEXT: ret i64 [[RESULT]]
  return __mperf();
}

unsigned long long test_aperf() {
  // CHECK-LABEL: test_aperf
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.x86.rdpru(i32 1)
  // CHECK-NEXT: ret i64 [[RESULT]]
  return __aperf();
}

void test_direct_calls_to_builtin_rdpru(int regid) {
  // CHECK: call i64 @llvm.x86.rdpru(i32 0)
  // CHECK: call i64 @llvm.x86.rdpru(i32 1)
  // CHECK: call i64 @llvm.x86.rdpru(i32 %{{.*}})
  (void) __builtin_ia32_rdpru(0);
  (void) __builtin_ia32_rdpru(1);
  (void) __builtin_ia32_rdpru(regid);
}
