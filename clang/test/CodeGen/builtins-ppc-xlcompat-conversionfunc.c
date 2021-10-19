// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern double da;
double test_fcfid() {
  // CHECK-LABEL: test_fcfid
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fcfid(double %0)
  return __builtin_ppc_fcfid(da);
}

double test_xl_fcfid() {
  // CHECK-LABEL: test_xl_fcfid
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fcfid(double %0)
  return __fcfid(da);
}

double test_fcfud() {
  // CHECK-LABEL: test_fcfud
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fcfud(double %0)
  return __builtin_ppc_fcfud(da);
}

double test_xl_fcfud() {
  // CHECK-LABEL: test_xl_fcfud
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fcfud(double %0)
  return __fcfud(da);
}

double test_fctid() {
  // CHECK-LABEL: test_fctid
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctid(double %0)
  return __builtin_ppc_fctid(da);
}

double test_xl_fctid() {
  // CHECK-LABEL: test_xl_fctid
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctid(double %0)
  return __fctid(da);
}

double test_fctidz() {
  // CHECK-LABEL: test_fctidz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctidz(double %0)
  return __builtin_ppc_fctidz(da);
}

double test_xl_fctidz() {
  // CHECK-LABEL: test_xl_fctidz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctidz(double %0)
  return __fctidz(da);
}

double test_fctiw() {
  // CHECK-LABEL: test_fctiw
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctiw(double %0)
  return __builtin_ppc_fctiw(da);
}

double test_xl_fctiw() {
  // CHECK-LABEL: test_xl_fctiw
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctiw(double %0)
  return __fctiw(da);
}

double test_fctiwz() {
  // CHECK-LABEL: test_fctiwz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctiwz(double %0)
  return __builtin_ppc_fctiwz(da);
}

double test_xl_fctiwz() {
  // CHECK-LABEL: test_xl_fctiwz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctiwz(double %0)
  return __fctiwz(da);
}

double test_fctudz() {
  // CHECK-LABEL: test_fctudz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctudz(double %0)
  return __builtin_ppc_fctudz(da);
}

double test_xl_fctudz() {
  // CHECK-LABEL: test_xl_fctudz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctudz(double %0)
  return __fctudz(da);
}

double test_fctuwz() {
  // CHECK-LABEL: test_fctuwz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctuwz(double %0)
  return __builtin_ppc_fctuwz(da);
}

double test_xl_fctuwz() {
  // CHECK-LABEL: test_xl_fctuwz
  // CHECK-NEXT: entry:
  // CHECK: double @llvm.ppc.fctuwz(double %0)
  return __fctuwz(da);
}
