// RUN: fir-opt --loop-versioning %s | FileCheck %s


//  subroutine sum1d(a, n)
//    real*8 :: a(:)
//    integer :: n
//    real*8 :: sum
//    integer :: i
//    sum = 0
//    do i=1,n
//       sum = sum + a(i)
//    end do
//  end subroutine sum1d
module {
  func.func @sum1d(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmoduleFsum1dEi"}
    %1 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QMmoduleFsum1dEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %1 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %2 = fir.convert %c1_i32 : (i32) -> index
    %3 = fir.load %arg1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %c1 = arith.constant 1 : index
    %5 = fir.convert %2 : (index) -> i32
    %6:2 = fir.do_loop %arg2 = %2 to %4 step %c1 iter_args(%arg3 = %5) -> (index, i32) {
      fir.store %arg3 to %0 : !fir.ref<i32>
      %7 = fir.load %1 : !fir.ref<f64>
      %8 = fir.load %0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %10 = arith.subi %9, %c1_i64 : i64
      %11 = fir.coordinate_of %arg0, %10 : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = arith.addf %7, %12 fastmath<contract> : f64
      fir.store %13 to %1 : !fir.ref<f64>
      %14 = arith.addi %arg2, %c1 : index
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 : i32
      fir.result %14, %17 : index, i32
    }
    fir.store %6#1 to %0 : !fir.ref<i32>
    return
  }

// Note this only checks the expected transformation, not the entire generated code:
// CHECK-LABEL: func.func @sum1d(
// CHECK-SAME:                  %[[ARG0:.*]]: !fir.box<!fir.array<?xf64>> {{.*}})
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[ZERO]] : {{.*}}
// CHECK: %[[SIZE:.*]] = arith.constant 8 : index
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[DIMS]]#2, %[[SIZE]]
// CHECK: %[[IF_RES:.*]]:2 = fir.if %[[CMP]] -> {{.*}}
// CHECK: %[[NEWARR:.*]] = fir.convert %[[ARG0]]
// CHECK: %[[BOXADDR:.*]] = fir.box_addr %[[NEWARR]] : {{.*}} -> !fir.ref<!fir.array<?xf64>>
// CHECK: %[[LOOP_RES:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[BOXADDR]], %{{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK: %{{.*}} = fir.load %[[COORD]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK  fir.result %[[LOOP_RES]]#0, %[[LOOP_RES]]#1
// CHECK: } else {
// CHECK: %[[LOOP_RES2:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD2:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
// CHECK: %{{.*}}= fir.load %[[COORD2]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK  fir.result %[[LOOP_RES2]]#0, %[[LOOP_RES2]]#1
// CHECK: }
// CHECK: fir.store %[[IF_RES]]#1 to %{{.*}}
// CHECK: return

// -----

// Test that loop-versioning pass doesn't expand known size arrays.
func.func @sum1dfixed(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum1dfixedEi"}
    %1 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum1dfixedEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %1 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %2 = fir.convert %c1_i32 : (i32) -> index
    %3 = fir.load %arg1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %c1 = arith.constant 1 : index
    %5 = fir.convert %2 : (index) -> i32
    %6:2 = fir.do_loop %arg2 = %2 to %4 step %c1 iter_args(%arg3 = %5) -> (index, i32) {
      fir.store %arg3 to %0 : !fir.ref<i32>
      %7 = fir.load %1 : !fir.ref<f64>
      %8 = fir.load %0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %10 = arith.subi %9, %c1_i64 : i64
      %11 = fir.coordinate_of %arg0, %10 : (!fir.ref<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = arith.addf %7, %12 fastmath<contract> : f64
      fir.store %13 to %1 : !fir.ref<f64>
      %14 = arith.addi %arg2, %c1 : index
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 : i32
      fir.result %14, %17 : index, i32
    }
    fir.store %6#1 to %0 : !fir.ref<i32>
    return
  }

// CHECK-LABEL: func.func @sum1dfixed(
// CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<!fir.array<?xf64>> {{.*}})
// CHECK: fir.do_loop {{.*}}
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG0]], {{.*}}
// CHECK: %{{.*}} = fir.load %[[COORD]]

} // End module
