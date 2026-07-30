// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:  FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | \
// RUN:  FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:  FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:  FileCheck %s

extern int ia, ib;

// td
void test_tw(void) {
// CHECK: void @llvm.ppc.tw(i32 %0, i32 %1, i32 1)

  __builtin_ppc_tw(ia, ib, 1);
}

void test_xl_tw(void) {
// CHECK: void @llvm.ppc.tw(i32 %0, i32 %1, i32 1)

  __tw(ia, ib, 1);
}

// trap
void test_trap(void) {
// CHECK: void @llvm.ppc.trap(i32 %0)
  __builtin_ppc_trap(ia);
}

void test_xl_trap(void) {
// CHECK: void @llvm.ppc.trap(i32 %0)
  __trap(ia);
}
