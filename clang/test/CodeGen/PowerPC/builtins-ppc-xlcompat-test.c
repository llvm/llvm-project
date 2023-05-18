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

int test_builtin_ppc_test() {
// CHECK-LABEL: @test_builtin_ppc_test
// CHECK: call i32 @llvm.ppc.compare.exp.uo(double %0, double %1)
// CHECK: call i32 @llvm.ppc.compare.exp.lt(double %3, double %4)
// CHECK: call i32 @llvm.ppc.compare.exp.gt(double %6, double %7)
// CHECK: call i32 @llvm.ppc.compare.exp.eq(double %9, double %10)
// CHECK: call i32 @llvm.ppc.test.data.class.f64(double %12, i32 0)
// CHECK: call i32 @llvm.ppc.test.data.class.f32(float %13, i32 0)
// CHECK: call i32 @llvm.ppc.compare.exp.uo(double %14, double %15)
// CHECK: call i32 @llvm.ppc.compare.exp.lt(double %17, double %18)
// CHECK: call i32 @llvm.ppc.compare.exp.gt(double %20, double %21)
// CHECK: call i32 @llvm.ppc.compare.exp.eq(double %23, double %24)
// CHECK: call i32 @llvm.ppc.test.data.class.f64(double %26, i32 127)
// CHECK: call i32 @llvm.ppc.test.data.class.f32(float %27, i32 127)

// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_uo' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_lt' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_gt' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_eq' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_uo' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_lt' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_gt' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_compare_exp_eq' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions
// CHECK-NONPWR9-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions

// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_uo' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_lt' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_gt' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_eq' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_uo' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_lt' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_gt' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_compare_exp_eq' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions,vsx
// CHECK-NOVSX-ERR: error: '__builtin_ppc_test_data_class' needs target feature isa-v30-instructions,vsx
  int i;
  i = __builtin_ppc_compare_exp_uo(d, d);
  i = __builtin_ppc_compare_exp_lt(d, d);
  i = __builtin_ppc_compare_exp_gt(d, d);
  i = __builtin_ppc_compare_exp_eq(d, d);
  i = __builtin_ppc_test_data_class(d, 0);
  i = __builtin_ppc_test_data_class(f, 0);
  i = __compare_exp_uo(d, d);
  i = __compare_exp_lt(d, d);
  i = __compare_exp_gt(d, d);
  i = __compare_exp_eq(d, d);
  i = __test_data_class(d, 127);
  i = __test_data_class(f, 127);
  return i;
}
