! Fails until we update the pass to use the `fir.do_concurrent` op.
! XFAIL: *

! Tests mapping of a `do concurrent` loop with multiple iteration ranges.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %t/multi_range.f90 -o - \
! RUN:   | FileCheck %s

!--- multi_range.f90
program main
   integer, parameter :: n = 20
   integer, parameter :: m = 40
   integer, parameter :: l = 60
   integer :: a(n, m, l)

   do concurrent(i=3:n, j=5:m, k=7:l)
       a(i,j,k) = i * j + k
   end do
end

! CHECK: func.func @_QQmain

! CHECK: %[[C3:.*]] = arith.constant 3 : i32
! CHECK: %[[LB_I:.*]] = fir.convert %[[C3]] : (i32) -> index
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: %[[UB_I:.*]] = fir.convert %[[C20]] : (i32) -> index
! CHECK: %[[STEP_I:.*]] = arith.constant 1 : index

! CHECK: %[[C5:.*]] = arith.constant 5 : i32
! CHECK: %[[LB_J:.*]] = fir.convert %[[C5]] : (i32) -> index
! CHECK: %[[C40:.*]] = arith.constant 40 : i32
! CHECK: %[[UB_J:.*]] = fir.convert %[[C40]] : (i32) -> index
! CHECK: %[[STEP_J:.*]] = arith.constant 1 : index

! CHECK: %[[C7:.*]] = arith.constant 7 : i32
! CHECK: %[[LB_K:.*]] = fir.convert %[[C7]] : (i32) -> index
! CHECK: %[[C60:.*]] = arith.constant 60 : i32
! CHECK: %[[UB_K:.*]] = fir.convert %[[C60]] : (i32) -> index
! CHECK: %[[STEP_K:.*]] = arith.constant 1 : index

! CHECK: omp.parallel {

! CHECK-NEXT: %[[ITER_VAR_I:.*]] = fir.alloca i32 {bindc_name = "i"}
! CHECK-NEXT: %[[BINDING_I:.*]]:2 = hlfir.declare %[[ITER_VAR_I]] {uniq_name = "_QFEi"}

! CHECK-NEXT: %[[ITER_VAR_J:.*]] = fir.alloca i32 {bindc_name = "j"}
! CHECK-NEXT: %[[BINDING_J:.*]]:2 = hlfir.declare %[[ITER_VAR_J]] {uniq_name = "_QFEj"}

! CHECK-NEXT: %[[ITER_VAR_K:.*]] = fir.alloca i32 {bindc_name = "k"}
! CHECK-NEXT: %[[BINDING_K:.*]]:2 = hlfir.declare %[[ITER_VAR_K]] {uniq_name = "_QFEk"}

! CHECK: omp.wsloop {
! CHECK-NEXT: omp.loop_nest
! CHECK-SAME:   (%[[ARG0:[^[:space:]]+]], %[[ARG1:[^[:space:]]+]], %[[ARG2:[^[:space:]]+]])
! CHECK-SAME:   : index = (%[[LB_I]], %[[LB_J]], %[[LB_K]])
! CHECK-SAME:     to (%[[UB_I]], %[[UB_J]], %[[UB_K]]) inclusive
! CHECK-SAME:     step (%[[STEP_I]], %[[STEP_J]], %[[STEP_K]]) {

! CHECK-NEXT: %[[IV_IDX_I:.*]] = fir.convert %[[ARG0]]
! CHECK-NEXT: fir.store %[[IV_IDX_I]] to %[[BINDING_I]]#0

! CHECK-NEXT: %[[IV_IDX_J:.*]] = fir.convert %[[ARG1]]
! CHECK-NEXT: fir.store %[[IV_IDX_J]] to %[[BINDING_J]]#0

! CHECK-NEXT: %[[IV_IDX_K:.*]] = fir.convert %[[ARG2]]
! CHECK-NEXT: fir.store %[[IV_IDX_K]] to %[[BINDING_K]]#0

! CHECK:      omp.yield
! CHECK-NEXT: }
! CHECK-NEXT: }

! CHECK-NEXT: omp.terminator
! CHECK-NEXT: }
