// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern void *vpa;

void test_dcbtstt(void) {
  // CHECK-LABEL: @test_dcbtstt
  // CHECK: %0 = load ptr, ptr @vpa
  // CHECK: tail call void @llvm.ppc.dcbtstt(ptr %0)
  // CHECK: ret void
  __dcbtstt(vpa);
}

void test_dcbtt(void) {
  // CHECK-LABEL: @test_dcbt
  // CHECK: %0 = load ptr, ptr @vpa
  // CHECK: tail call void @llvm.ppc.dcbtt(ptr %0)
  // CHECK: ret void
  __dcbtt(vpa);
}
