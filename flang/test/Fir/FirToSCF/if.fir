
// RUN: fir-opt %s --fir-to-scf | FileCheck %s

// CHECK: func.func @_QFPtest_only(
// CHECK:   %[[ARG0:.*]]: !fir.ref<tuple<!fir.ref<f32>>>) {
// CHECK:   %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:   %[[VAL_1:.*]] = arith.constant false
// CHECK:   %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:   scf.if %[[VAL_1:.*]] {
// CHECK:    %[[VAL_3:.*]] = arith.addi %[[VAL_2:.*]], %[[VAL_0:.*]] : i32
// CHECK:   }
// CHECK:   return
// CHECK:   }
func.func @_QFPtest_only(%arg0: !fir.ref<tuple<!fir.ref<f32>>>) {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    fir.if %false {
      %0 = arith.addi %c0_i32, %c1_i32 : i32
    }
    return
  }

// CHECK: func.func @_QFPtest_else(
// CHECK:   %[[ARG0:.*]]: !fir.ref<tuple<!fir.ref<f32>>>) {
// CHECK:   %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:   %[[VAL_1:.*]] = arith.constant false
// CHECK:   %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:   %[[VAL_4:.*]] = fir.coordinate_of %[[ARG0:.*]], %[[VAL_2:.*]] : (!fir.ref<tuple<!fir.ref<f32>>>, i32) -> !fir.llvm_ptr<!fir.ref<f32>>
// CHECK:   %[[VAL_5:.*]] = fir.load %[[VAL_4:.*]] : !fir.llvm_ptr<!fir.ref<f32>>
// CHECK:   %[[VAL_6:.*]] = fir.declare %[[VAL_5:.*]] {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:   %[[VAL_7:.*]] = fir.address_of(@_QFFtest_elseEsum) : !fir.ref<i32>
// CHECK:   %[[VAL_10:.*]] = fir.declare %[[VAL_7:.*]] {uniq_name = "_QFFtest_elseEsum"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:   scf.if %[[VAL_1:.*]] {
// CHECK:     %[[VAL_8:.*]] = fir.load %[[VAL_10:.*]] : !fir.ref<i32>
// CHECK:     %[[VAL_9:.*]] = arith.addi %[[VAL_8:.*]], %[[VAL_0:.*]] : i32
// CHECK:     fir.store %[[VAL_9:.*]] to %[[VAL_10:.*]] : !fir.ref<i32>
// CHECK:   } else {
// CHECK:     %[[VAL_8:.*]] = fir.load %[[VAL_10:.*]] : !fir.ref<i32>
// CHECK:     %[[VAL_9:.*]] = arith.addi %[[VAL_8:.*]], %c1_i32 : i32
// CHECK:     fir.store %[[VAL_9:.*]] to %[[VAL_10:.*]] : !fir.ref<i32>
// CHECK:   }
// CHECK:   return
// CHECK: }
func.func  @_QFPtest_else(%arg0: !fir.ref<tuple<!fir.ref<f32>>> {}) attributes {} {
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.ref<f32>>>, i32) -> !fir.llvm_ptr<!fir.ref<f32>>
  %2 = fir.load %1 : !fir.llvm_ptr<!fir.ref<f32>>
  %3 = fir.declare %2 {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.address_of(@_QFFtest_elseEsum) : !fir.ref<i32>
  %5 = fir.declare %4 {uniq_name = "_QFFtest_elseEsum"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.if %false {
    %6 = fir.load %5 : !fir.ref<i32>
    %7 = arith.addi %6, %c1_i32 : i32
    fir.store %7 to %5 : !fir.ref<i32>
  } else {
    %6 = fir.load %5 : !fir.ref<i32>
    %7 = arith.addi %6, %c1_i32 : i32
    fir.store %7 to %5 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL:     func.func @test_two_result() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_0:.*]], %[[VAL_1:.*]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 3.140000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2.710000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[RES:[0-9]+]]:2 = scf.if %[[VAL_2:.*]] -> (f32, f32) {
// CHECK:            scf.yield %[[VAL_3:.*]], %[[VAL_4:.*]] : f32, f32
// CHECK:            } else {
// CHECK:              scf.yield %[[VAL_5:.*]], %[[VAL_6:.*]] : f32, f32
// CHECK:            }
// CHECK:            return 
// CHECK:            }
func.func @test_two_result() {
  %c10_i32 = arith.constant 10 : i32
  %c5_i32 = arith.constant 5 : i32
  %cmp = arith.cmpi sgt, %c10_i32, %c5_i32 : i32
  %c3_14_f32 = arith.constant 3.14 : f32
  %c2_71_f32 = arith.constant 2.71 : f32
  %c1_0_f32 = arith.constant 1.0 : f32
  %c2_0_f32 = arith.constant 2.0 : f32
  %x, %y = fir.if %cmp -> (f32, f32) {
    fir.result %c3_14_f32, %c2_71_f32 : f32, f32
  } else {
    fir.result %c1_0_f32, %c2_0_f32 : f32, f32
  }
  return
}
