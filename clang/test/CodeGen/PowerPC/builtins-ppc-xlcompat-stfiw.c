// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s

extern const int *cia;
extern double da;

void test_stfiw(void) {
  // CHECK-LABEL: test_stfiw
  // CHECK: void @llvm.ppc.stfiw(i8* %0, double %1)
  __builtin_ppc_stfiw(cia, da);
}

void test_xl_stfiw(void) {
  // CHECK-LABEL: test_xl_stfiw
  // CHECK: void @llvm.ppc.stfiw(i8* %0, double %1)
  __stfiw(cia, da);
}
