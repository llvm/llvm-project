// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR9-ERR
// RUN: not %clang_cc1 -target-feature -vsx -target-cpu pwr9 \
// RUN:   -triple powerpc64-unknown-linux-gnu -emit-llvm-only %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOVSX-ERR

extern double d;
extern float f;

int test_builtin_ppc_compare_exp_uo() {
// CHECK-LABEL:       @test_builtin_ppc_compare_exp_uo
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.uo(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_compare_exp_uo(d, d);
}

int test_builtin_ppc_compare_exp_lt() {
// CHECK-LABEL:       @test_builtin_ppc_compare_exp_lt
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.lt(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_compare_exp_lt(d, d);
}

int test_builtin_ppc_compare_exp_gt() {
// CHECK-LABEL:       @test_builtin_ppc_compare_exp_gt
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.gt(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_compare_exp_gt(d, d);
}

int test_builtin_ppc_compare_exp_eq() {
// CHECK-LABEL:       @test_builtin_ppc_compare_exp_eq
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.eq(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_compare_exp_eq(d, d);
}

int test_builtin_ppc_test_data_class_d() {
// CHECK-LABEL:       @test_builtin_ppc_test_data_class_d
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.test.data.class.d(double %0, i32 0)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_test_data_class(d, 0);
}

int test_builtin_ppc_test_data_class_f() {
// CHECK-LABEL:       @test_builtin_ppc_test_data_class_f
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.test.data.class.f(float %0, i32 0)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __builtin_ppc_test_data_class(f, 0);
}

int test_compare_exp_uo() {
// CHECK-LABEL:       @test_compare_exp_uo
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.uo(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __compare_exp_uo(d, d);
}

int test_compare_exp_lt() {
// CHECK-LABEL:       @test_compare_exp_lt
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.lt(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __compare_exp_lt(d, d);
}

int test_compare_exp_gt() {
// CHECK-LABEL:       @test_compare_exp_gt
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.gt(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __compare_exp_gt(d, d);
}

int test_compare_exp_eq() {
// CHECK-LABEL:       @test_compare_exp_eq
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.compare.exp.eq(double %0, double %1)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __compare_exp_eq(d, d);
}

int test_test_data_class_d() {
// CHECK-LABEL:       @test_test_data_class_d
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.test.data.class.d(double %0, i32 127)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __test_data_class(d, 127);
}

int test_test_data_class_f() {
// CHECK-LABEL:       @test_test_data_class_f
// CHECK:             [[TMP:%.*]] = call i32 @llvm.ppc.test.data.class.f(float %0, i32 127)
// CHECK-NEXT:        ret i32 [[TMP]]
// CHECK-NONPWR9-ERR: error: this builtin is only valid on POWER9 or later CPUs
// CHECK-NOVSX-ERR: error: this builtin requires VSX to be enabled
  return __test_data_class(f, 127);
}
