// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:   FileCheck %s --check-prefix=CHECK-32BIT
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern unsigned long ula;

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

void test_mtmsr(void) {
  // CHECK-LABEL: @test_mtmsr
  // CHECK: tail call void @llvm.ppc.mtmsr(i32 %conv)
  // CHECK-32BIT-LABEL: @test_mtmsr
  // CHECK-32BIT: tail call void @llvm.ppc.mtmsr(i32 %0)
  __mtmsr(ula);
}

unsigned long test_mfspr(void) {
  // CHECK-LABEL: @test_mfspr
  // CHECK: %0 = tail call i64 @llvm.ppc.mfspr.i64(i32 898)
  return __mfspr(898);
}

void test_mtspr(void) {
  // CHECK-LABEL: @test_mtspr
  // CHECK: tail call void @llvm.ppc.mtspr.i64(i32 1, i64 %0)
  __mtspr(1, ula);
}
