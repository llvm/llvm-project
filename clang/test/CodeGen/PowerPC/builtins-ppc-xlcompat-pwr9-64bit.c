// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr9 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR9-ERR

extern signed long long sll;
extern unsigned long long ull;

signed long long test_builtin_ppc_cmpeqb() {
  // CHECK-LABEL:    @test_builtin_ppc_cmpeqb(
  // CHECK:          %2 = call i64 @llvm.ppc.cmpeqb(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_cmpeqb(sll, sll);
}

long long test_builtin_ppc_setb() {
  // CHECK-LABEL: @test_builtin_ppc_setb(
  // CHECK:       %2 = call i64 @llvm.ppc.setb(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_setb(sll, sll);
}

signed long long test_builtin_ppc_maddhd() {
  // CHECK-LABEL:    @test_builtin_ppc_maddhd(
  // CHECK:          %3 = call i64 @llvm.ppc.maddhd(i64 %0, i64 %1, i64 %2)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_maddhd(sll, sll, sll);
}

unsigned long long test_builtin_ppc_maddhdu() {
  // CHECK-LABEL:    @test_builtin_ppc_maddhdu(
  // CHECK:          %3 = call i64 @llvm.ppc.maddhdu(i64 %0, i64 %1, i64 %2)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_maddhdu(ull, ull, ull);
}

signed long long test_builtin_ppc_maddld() {
  // CHECK-LABEL:    @test_builtin_ppc_maddld(
  // CHECK:          %3 = call i64 @llvm.ppc.maddld(i64 %0, i64 %1, i64 %2)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_maddld(sll, sll, sll);
}

unsigned long long test_builtin_ppc_maddld_unsigned() {
  // CHECK-LABEL:    @test_builtin_ppc_maddld_unsigned(
  // CHECK:          %3 = call i64 @llvm.ppc.maddld(i64 %0, i64 %1, i64 %2)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_maddld(ull, ull, ull);
}

unsigned long long extract_sig (double d) {
// CHECK-LABEL: @extract_sig(
// CHECK:       [[TMP1:%.*]] = call i64 @llvm.ppc.extract.sig(double %0)
// CHECK-NEXT:  ret i64 [[TMP1]]
//
// CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
// CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __extract_sig (d);
}

double insert_exp (double d, unsigned long long ull) {
// CHECK-LABEL: @insert_exp(
// CHECK:       [[TMP2:%.*]] = call double @llvm.ppc.insert.exp(double %0, i64 %1)
// CHECK-NEXT:    ret double [[TMP2]]
//
// CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
// CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __insert_exp (d, ull);
}

signed long long test_builtin_ppc_addex0() {
  // CHECK-LABEL:    @test_builtin_ppc_addex0
  // CHECK:          %2 = call i64 @llvm.ppc.addex(i64 %0, i64 %1, i32 0)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_addex(sll, sll, 0);
}

unsigned long long test_builtin_ppc_addex1() {
  // CHECK-LABEL:    @test_builtin_ppc_addex1
  // CHECK:          %2 = call i64 @llvm.ppc.addex(i64 %0, i64 %1, i32 0)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_addex(ull, ull, 0);
}
