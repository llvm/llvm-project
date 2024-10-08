// RUN: fir-opt --constant-argument-globalisation-opt  < %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm -flang-deprecated-no-hlfir -O2 -o - %s | FileCheck --check-prefix=DISABLE %s
module {
  func.func @sub1(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "y"}) {
    %0 = fir.alloca i32 {adapt.valuebyref}
    %1 = fir.alloca f64 {adapt.valuebyref}
    %2 = fir.alloca f64 {adapt.valuebyref}
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %3 = fir.declare %arg0 {uniq_name = "_QFsub1Ex"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.declare %arg1 {uniq_name = "_QFsub1Ey"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.store %cst_0 to %2 : !fir.ref<f64>
    %false = arith.constant false
    fir.store %cst to %1 : !fir.ref<f64>
    %false_1 = arith.constant false
    fir.store %c1_i32 to %0 : !fir.ref<i32>
    %false_2 = arith.constant false
    fir.call @sub2(%2, %1, %3, %4, %0) fastmath<contract> : (!fir.ref<f64>, !fir.ref<f64>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) -> ()
    return
  }
  func.func private @sub2(!fir.ref<f64>, !fir.ref<f64>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)

// CHECK-LABEL: func.func @sub1(
// CHECK-SAME: [[ARG0:%.*]]: !fir.ref<i32> {{{.*}}},
// CHECK-SAME: [[ARG1:%.*]]: !fir.ref<i32> {{{.*}}}) {
// CHECK: [[X:%.*]] = fir.declare [[ARG0]] {{.*}}
// CHECK: [[Y:%.*]] = fir.declare [[ARG1]] {{.*}}
// CHECK: [[CONST_R0:%.*]] = fir.address_of([[EXTR_0:@.*]]) : !fir.ref<f64>
// CHECK: [[CONST_R1:%.*]] = fir.address_of([[EXTR_1:@.*]]) : !fir.ref<f64>
// CHECK: [[CONST_I:%.*]] = fir.address_of([[EXTR_2:@.*]]) : !fir.ref<i32>
// CHECK: fir.call @sub2([[CONST_R0]], [[CONST_R1]], [[X]], [[Y]], [[CONST_I]])
// CHECK-SAME: fastmath<contract>
// CHECK: return

// CHECK: fir.global internal [[EXTR_0]] constant : f64 {
// CHECK: %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK: fir.has_value %{{.*}} : f64
// CHECK: }
// CHECK: fir.global internal [[EXTR_1]] constant : f64 {
// CHECK: %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK: fir.has_value %{{.*}} : f64
// CHECK: }
// CHECK: fir.global internal [[EXTR_2]] constant : i32 {
// CHECK: %{{.*}} = arith.constant 1 : i32
// CHECK: fir.has_value %{{.*}} : i32
// CHECK: }

// DISABLE-LABEL: ; ModuleID =
// DISABLE-NOT: @_extruded
// DISABLE:  define void @sub1(
// DISABLE-SAME: ptr [[ARG0:%.*]],
// DISABLE-SAME: ptr [[ARG1:%.*]])
// DISABLE-SAME: {
// DISABLE: [[CONST_R0:%.*]] = alloca double
// DISABLE: [[CONST_R1:%.*]] = alloca double
// DISABLE: [[CONST_I:%.*]] = alloca i32
// DISABLE: store double 0.0{{.*}}+00, ptr [[CONST_R0]]
// DISABLE: store double 1.0{{.*}}+00, ptr [[CONST_R1]]
// DISABLE: store i32 1, ptr [[CONST_I]]
// DISABLE: call void @sub2(ptr nonnull [[CONST_R0]],
// DISABLE-SAME: ptr nonnull [[CONST_R1]], 
// DISABLE-SAME: ptr [[ARG0]], ptr [[ARG1]],
// DISABLE-SAME: ptr nonnull [[CONST_I]])
// DISABLE: ret void
// DISABLE: }
}
