!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Check that the presence tests are done outside of the atomic update
! construct.

!CHECK-LABEL: func.func @_QPf00
!CHECK: %[[VAL_A:[0-9]+]]:2 = hlfir.declare %arg0 dummy_scope %0
!CHECK: %[[VAL_X:[0-9]+]]:2 = hlfir.declare %arg1 dummy_scope %0
!CHECK: %[[VAL_Y:[0-9]+]]:2 = hlfir.declare %arg2 dummy_scope %0
!CHECK: %[[V4:[0-9]+]] = fir.load %[[VAL_X]]#0 : !fir.ref<f32>
!CHECK: %[[V5:[0-9]+]] = fir.load %[[VAL_X]]#0 : !fir.ref<f32>
!CHECK: %[[V6:[0-9]+]] = fir.is_present %[[VAL_Y]]#0 : (!fir.ref<f32>) -> i1
!CHECK: %[[V7:[0-9]+]] = arith.cmpf ogt, %[[V4]], %[[V5]] fastmath<contract> : f32
!CHECK: %[[V8:[0-9]+]] = arith.select %[[V7]], %[[V4]], %[[V5]] : f32
!CHECK: %[[V9:[0-9]+]] = fir.if %[[V6]] -> (f32) {
!CHECK:   %[[V10:[0-9]+]] = fir.load %[[VAL_Y]]#0 : !fir.ref<f32>
!CHECK:   %[[V11:[0-9]+]] = arith.cmpf ogt, %[[V8]], %[[V10]] fastmath<contract> : f32
!CHECK:   %[[V12:[0-9]+]] = arith.select %[[V11]], %[[V8]], %[[V10]] : f32
!CHECK:   fir.result %[[V12]] : f32
!CHECK: } else {
!CHECK:   fir.result %[[V8]] : f32
!CHECK: }
!CHECK: omp.atomic.update memory_order(relaxed) %[[VAL_A]]#0 : !fir.ref<f32> {
!CHECK: ^bb0(%[[ARG:[a-z0-9]+]]: f32):
!CHECK:   %[[V10:[0-9]+]] = arith.cmpf ogt, %[[ARG]], %[[V9]] fastmath<contract> : f32
!CHECK:   %[[V11:[0-9]+]] = arith.select %[[V10]], %[[ARG]], %[[V9]] : f32
!CHECK:   omp.yield(%[[V11]] : f32)
!CHECK: }

subroutine f00(a, x, y)
  real :: a
  real, optional :: x, y
  !$omp atomic update
  a = max(x, a, y)
end


!CHECK-LABEL: func.func @_QPf01
!CHECK: %[[VAL_A:[0-9]+]]:2 = hlfir.declare %arg0 dummy_scope %0
!CHECK: %[[VAL_X:[0-9]+]]:2 = hlfir.declare %arg1 dummy_scope %0
!CHECK: %[[VAL_Y:[0-9]+]]:2 = hlfir.declare %arg2 dummy_scope %0
!CHECK: %[[V4:[0-9]+]] = fir.load %[[VAL_X]]#0 : !fir.ref<i32>
!CHECK: %[[V5:[0-9]+]] = fir.load %[[VAL_X]]#0 : !fir.ref<i32>
!CHECK: %[[V6:[0-9]+]] = fir.is_present %[[VAL_Y]]#0 : (!fir.ref<i32>) -> i1
!CHECK: %[[V7:[0-9]+]] = arith.cmpi slt, %[[V4]], %[[V5]] : i32
!CHECK: %[[V8:[0-9]+]] = arith.select %[[V7]], %[[V4]], %[[V5]] : i32
!CHECK: %[[V9:[0-9]+]] = fir.if %[[V6]] -> (i32) {
!CHECK:   %[[V10:[0-9]+]] = fir.load %[[VAL_Y]]#0 : !fir.ref<i32>
!CHECK:   %[[V11:[0-9]+]] = arith.cmpi slt, %[[V8]], %[[V10]] : i32
!CHECK:   %[[V12:[0-9]+]] = arith.select %[[V11]], %[[V8]], %[[V10]] : i32
!CHECK:   fir.result %[[V12]] : i32
!CHECK: } else {
!CHECK:   fir.result %[[V8]] : i32
!CHECK: }
!CHECK: omp.atomic.update memory_order(relaxed) %[[VAL_A]]#0 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:[a-z0-9]+]]: i32):
!CHECK:   %[[V10:[0-9]+]] = arith.cmpi slt, %[[ARG]], %[[V9]] : i32
!CHECK:   %[[V11:[0-9]+]] = arith.select %[[V10]], %[[ARG]], %[[V9]] : i32
!CHECK:   omp.yield(%[[V11]] : i32)
!CHECK: }

subroutine f01(a, x, y)
  integer :: a
  integer, optional :: x, y
  !$omp atomic update
  a = min(x, a, y)
end

