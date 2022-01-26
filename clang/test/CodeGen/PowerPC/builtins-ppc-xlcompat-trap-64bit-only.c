// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:  FileCheck %s --check-prefixes=CHECK64
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | \
// RUN:  FileCheck %s --check-prefixes=CHECK64
// RUN: not %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 2>&1 | \
// RUN:  FileCheck %s -check-prefixes=CHECK32-ERROR
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | \
// RUN:  FileCheck %s --check-prefixes=CHECK64

extern long long lla, llb;
extern double da;

// tdw
void test_xl_tdw(void) {
// CHECK64: void @llvm.ppc.tdw(i64 %0, i64 %1, i32 1)
// CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  __tdw(lla, llb, 1);
}

void test_tdw(void) {
// CHECK64: void @llvm.ppc.tdw(i64 %0, i64 %1, i32 13)
// CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  __builtin_ppc_tdw(lla, llb, 13);
}

// trapd
void test_trapd(void) {
// CHECK64: void @llvm.ppc.trapd(i64 %conv)
// CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  __builtin_ppc_trapd(da);
}

void test_xl_trapd(void) {
// CHECK64: void @llvm.ppc.trapd(i64 %conv)
// CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  __trapd(da);
}
