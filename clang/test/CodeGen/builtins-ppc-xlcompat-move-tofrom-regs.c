// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

unsigned int test_mftbu(void) {
  // CHECK-LABEL: @test_mftbu
  // CHECK: %0 = tail call i32 @llvm.ppc.mftbu()
  return __mftbu();
}

unsigned long test_mfmsr(void) {
  // CHECK-LABEL: @test_mfmsr
  // CHECK: %0 = tail call i32 @llvm.ppc.mfmsr()
  return __mfmsr();
}
