! Tests mapping of a `do concurrent` loop with multiple iteration ranges.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %t/multi_range.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %t/multi_range.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

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

! COMMON: func.func @_QQmain

! COMMON: %[[C3:.*]] = arith.constant 3 : i32
! COMMON: %[[LB_I:.*]] = fir.convert %[[C3]] : (i32) -> index
! COMMON: %[[C20:.*]] = arith.constant 20 : i32
! COMMON: %[[UB_I:.*]] = fir.convert %[[C20]] : (i32) -> index
! COMMON: %[[STEP_I:.*]] = arith.constant 1 : index

! COMMON: %[[C5:.*]] = arith.constant 5 : i32
! COMMON: %[[LB_J:.*]] = fir.convert %[[C5]] : (i32) -> index
! COMMON: %[[C40:.*]] = arith.constant 40 : i32
! COMMON: %[[UB_J:.*]] = fir.convert %[[C40]] : (i32) -> index
! COMMON: %[[STEP_J:.*]] = arith.constant 1 : index

! COMMON: %[[C7:.*]] = arith.constant 7 : i32
! COMMON: %[[LB_K:.*]] = fir.convert %[[C7]] : (i32) -> index
! COMMON: %[[C60:.*]] = arith.constant 60 : i32
! COMMON: %[[UB_K:.*]] = fir.convert %[[C60]] : (i32) -> index
! COMMON: %[[STEP_K:.*]] = arith.constant 1 : index

! DEVICE: omp.target host_eval(
! DEVICE-SAME: %[[LB_I]] -> %[[LB_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[UB_I]] -> %[[UB_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[STEP_I]] -> %[[STEP_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[LB_J]] -> %[[LB_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[UB_J]] -> %[[UB_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[STEP_J]] -> %[[STEP_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[LB_K]] -> %[[LB_K:[[:alnum:]]+]],
! DEVICE-SAME: %[[UB_K]] -> %[[UB_K:[[:alnum:]]+]],
! DEVICE-SAME: %[[STEP_K]] -> %[[STEP_K:[[:alnum:]]+]] :
! DEVICE-SAME: index, index, index, index, index, index, index, index, index)

! DEVICE: omp.teams

! HOST-NOT: omp.target
! HOST-NOT: omp.teams

! COMMON: omp.parallel {

! COMMON-NEXT: %[[ITER_VAR_I:.*]] = fir.alloca i32 {bindc_name = "i"}
! COMMON-NEXT: %[[BINDING_I:.*]]:2 = hlfir.declare %[[ITER_VAR_I]] {uniq_name = "_QFEi"}

! COMMON-NEXT: %[[ITER_VAR_J:.*]] = fir.alloca i32 {bindc_name = "j"}
! COMMON-NEXT: %[[BINDING_J:.*]]:2 = hlfir.declare %[[ITER_VAR_J]] {uniq_name = "_QFEj"}

! COMMON-NEXT: %[[ITER_VAR_K:.*]] = fir.alloca i32 {bindc_name = "k"}
! COMMON-NEXT: %[[BINDING_K:.*]]:2 = hlfir.declare %[[ITER_VAR_K]] {uniq_name = "_QFEk"}

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON-NEXT: omp.loop_nest
! COMMON-SAME:   (%[[ARG0:[^[:space:]]+]], %[[ARG1:[^[:space:]]+]], %[[ARG2:[^[:space:]]+]])
! COMMON-SAME:   : index = (%[[LB_I]], %[[LB_J]], %[[LB_K]])
! COMMON-SAME:     to (%[[UB_I]], %[[UB_J]], %[[UB_K]]) inclusive
! COMMON-SAME:     step (%[[STEP_I]], %[[STEP_J]], %[[STEP_K]]) {

! COMMON-NEXT: %[[IV_IDX_I:.*]] = fir.convert %[[ARG0]]
! COMMON-NEXT: fir.store %[[IV_IDX_I]] to %[[BINDING_I]]#0

! COMMON-NEXT: %[[IV_IDX_J:.*]] = fir.convert %[[ARG1]]
! COMMON-NEXT: fir.store %[[IV_IDX_J]] to %[[BINDING_J]]#0

! COMMON-NEXT: %[[IV_IDX_K:.*]] = fir.convert %[[ARG2]]
! COMMON-NEXT: fir.store %[[IV_IDX_K]] to %[[BINDING_K]]#0

! COMMON:      omp.yield
! COMMON-NEXT: }
! COMMON-NEXT: }

! HOST-NEXT: omp.terminator
! HOST-NEXT: }
