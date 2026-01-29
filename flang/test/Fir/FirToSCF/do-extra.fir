// Test conversions of fir.do_loop, including cases nested in other regions.
//
// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// Test that compiler generated fir.allocas are converted, slightly edited from:
// subroutine loop(a)
//   integer :: a(4)
//   a = 1
// end

// CHECK-LABEL: func.func @implied_do
// CHECK:       [[SUB:%[0-9]+]] = arith.subi %c4{{.*}}, %c1{{.*}} : index
// CHECK-NEXT:  [[ADD:%[0-9]+]] = arith.addi [[SUB]], %c1{{.*}} : index
// CHECK-NEXT:  [[DIV:%[0-9]+]] = arith.divsi [[ADD]], %c1{{.*}} : index
// CHECK:       scf.for %arg1 = %c0{{.*}} to [[DIV]] step %c1{{.*}} {

func.func @implied_do(%arg0: !fir.ref<!fir.array<4xi32>> {fir.bindc_name = "a"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c4 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFloopEa"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<4xi32>>
  fir.do_loop %arg1 = %c1 to %c4 step %c1 unordered {
    %3 = fir.array_coor %2(%1) %arg1 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %3 : !fir.ref<i32>
  }
  return
}

// Test fir.do_loop that is not directly inside a func.func but nested inside
// another region.
//
// subroutine implied_do_in_acc_serial(a)
//   integer :: a(4)
//   !$acc serial
//   a = 1
//   !$acc end serial
// end

// CHECK-LABEL: @implied_do_in_acc_serial
// CHECK-NOT:   fir.do_loop
// CHECK:       acc.serial
// CHECK:       scf.for
// CHECK:       acc.yield

func.func @implied_do_in_acc_serial(%arg0: !fir.ref<!fir.array<4xi32>> {fir.bindc_name = "a"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c4 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFloopEa"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<4xi32>>
  acc.serial {
    fir.do_loop %arg1 = %c1 to %c4 step %c1 unordered {
      %3 = fir.array_coor %2(%1) %arg1 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
      fir.store %c1_i32 to %3 : !fir.ref<i32>
    }
    acc.yield
  }
  return
}

// Test single result fir.do_loop conversion, edited down from:
// subroutine printImplied(a)
//   integer :: a(4)
//   write(*,*) (a(i),i=1,4)
// end

// CHECK-LABEL: func.func @_QPprintimplied
// CHECK:       [[SUB:%[0-9]+]] = arith.subi %c4, %c1 : index
// CHECK:       [[ADD:%[0-9]+]] = arith.addi [[SUB]], %c1 : index
// CHECK:       [[DIV:%[0-9]+]] = arith.divsi [[ADD]], %c1 : index
// CHECK:       [[FOR:%[0-9]+]] = scf.for %{{.*}} = %{{.*}} to [[DIV]] step %{{.*}} iter_args(%{{.*}} = %c1) -> (index) {
// CHECK:       [[IV:%[0-9]+]] = arith.addi %c1, {{.*}} : index
// CHECK:       fir.convert [[IV]] : (index) -> i32
// CHECK:       scf.yield

func.func @_QPprintimplied(%arg0: !fir.ref<!fir.array<4xi32>> {fir.bindc_name = "a"}) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c4 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFprintimpliedEa"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<4xi32>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFprintimpliedEi"}
  %4 = fir.declare %3 {uniq_name = "_QFprintimpliedEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.do_loop %arg1 = %c1 to %c4 step %c1 -> index {
    %13 = fir.convert %arg1 : (index) -> i32
    fir.store %13 to %4 : !fir.ref<i32>
    %14 = fir.load %4 : !fir.ref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = fir.array_coor %2(%1) %15 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %19 = arith.addi %arg1, %c1 : index
    fir.result %19 : index
  }
  %6 = fir.convert %5 : (index) -> i32
  fir.store %6 to %4 : !fir.ref<i32>
  return
}

// Test of a simple maxval do loop.
//
// CHECK-LABEL: func.func @mv_
// CHECK:       [[FOR:%[0-9]+]]:2 = scf.for
// CHECK:       fir.store [[FOR]]#0

func.func @mv_(%arg0: !fir.ref<!fir.array<3xi32>> {fir.bindc_name = "a", llvm.nocapture}, %arg1: !fir.ref<i32> {fir.bindc_name = "r", llvm.nocapture}) attributes {fir.internal_name = "_QPmv"} {
  %false = arith.constant false
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %c-2147483648_i32 = arith.constant -2147483648 : i32
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c3 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFmvEa"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
  %3 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFmvEr"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4:2 = fir.do_loop %arg2 = %c1 to %c3 step %c1 unordered iter_args(%arg3 = %c-2147483648_i32, %arg4 = %true) -> (i32, i1) {
    %5 = fir.array_coor %2(%1) %arg2 : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %6 = fir.load %5 : !fir.ref<i32>
    %7 = arith.cmpi sgt, %6, %arg3 : i32
    %8 = arith.ori %7, %arg4 : i1
    %9 = arith.select %8, %6, %arg3 : i32
    fir.result %9, %false : i32, i1
  }
  fir.store %4#0 to %3 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   func.func @do_with_iv_result_noinc(
// CHECK-SAME:        %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: !fir.ref<index>) {
// CHECK:           %[[VAL_0:.*]] = arith.subi %[[ARG1]], %[[ARG0]] : index
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %[[ARG2]] : index
// CHECK:           %[[VAL_2:.*]] = arith.divsi %[[VAL_1]], %[[ARG2]] : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[VAL_2]] step %[[C1]] iter_args(%{{.*}} = %[[ARG0]]) -> (index) {
// CHECK:             %[[MUL:.*]] = arith.muli %{{.*}}, %[[ARG2]] : index
// CHECK:             %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[MUL]] : index
// CHECK:             %[[ADD1:.*]] = arith.addi %[[ADD0]], %[[ARG2]] : index
// CHECK:             scf.yield %[[ADD1]] : index
// CHECK:           }

func.func @do_with_iv_result_noinc(%arg0: index, %arg1: index, %arg2: index, %arg3: !fir.ref<index>) {
  %0 = fir.do_loop %arg4 = %arg0 to %arg1 step %arg2 -> index {
    fir.result %arg4 : index
  }
  fir.store %0 to %arg3 : !fir.ref<index>
  return
}


