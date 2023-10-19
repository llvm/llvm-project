! Tests for 2.9.3.1 Simd

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsimdloop()
subroutine simdloop
integer :: i
  !$OMP SIMD
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = arith.constant 9 : i32
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.simdloop for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i=1, 9
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD 
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_if_clause
subroutine simdloop_with_if_clause(n, threshold)
integer :: i, n, threshold
  !$OMP SIMD IF( n .GE. threshold )
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sge
  ! CHECK: omp.simdloop if(%[[COND:.*]]) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_simdlen_clause
subroutine simdloop_with_simdlen_clause(n, threshold)
integer :: i, n, threshold
  !$OMP SIMD SIMDLEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop simdlen(2) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_simdlen_clause_from_param
subroutine simdloop_with_simdlen_clause_from_param(n, threshold)
integer :: i, n, threshold
integer, parameter :: simdlen = 2;
  !$OMP SIMD SIMDLEN(simdlen)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop simdlen(2) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_simdlen_clause_from_expr_from_param
subroutine simdloop_with_simdlen_clause_from_expr_from_param(n, threshold)
integer :: i, n, threshold
integer, parameter :: simdlen = 2;
  !$OMP SIMD SIMDLEN(simdlen*2 + 2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop simdlen(6) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_safelen_clause
subroutine simdloop_with_safelen_clause(n, threshold)
integer :: i, n, threshold
  !$OMP SIMD SAFELEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop safelen(2) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_safelen_clause_from_expr_from_param
subroutine simdloop_with_safelen_clause_from_expr_from_param(n, threshold)
integer :: i, n, threshold
integer, parameter :: safelen = 2;
  !$OMP SIMD SAFELEN(safelen*2 + 2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop safelen(6) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_simdlen_safelen_clause
subroutine simdloop_with_simdlen_safelen_clause(n, threshold)
integer :: i, n, threshold
  !$OMP SIMD SIMDLEN(1) SAFELEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %arg0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simdloop simdlen(1) safelen(2) for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive  step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_with_collapse_clause
subroutine simdloop_with_collapse_clause(n)
integer :: i, j, n
integer :: A(n,n)
! CHECK: %[[LOWER_I:.*]] = arith.constant 1 : i32
! CHECK: %[[UPPER_I:.*]] = fir.load %[[PARAM_ARG:.*]] : !fir.ref<i32>
! CHECK: %[[STEP_I:.*]] = arith.constant 1 : i32
! CHECK: %[[LOWER_J:.*]] = arith.constant 1 : i32
! CHECK: %[[UPPER_J:.*]] = fir.load %[[PARAM_ARG:.*]] : !fir.ref<i32>
! CHECK: %[[STEP_J:.*]] = arith.constant 1 : i32
! CHECK: omp.simdloop  for (%[[ARG_0:.*]], %[[ARG_1:.*]]) : i32 = (
! CHECK-SAME:               %[[LOWER_I]], %[[LOWER_J]]) to (
! CHECK-SAME:               %[[UPPER_I]], %[[UPPER_J]]) inclusive step (
! CHECK-SAME:               %[[STEP_I]], %[[STEP_J]]) {
  !$OMP SIMD COLLAPSE(2)
  do i = 1, n
    do j = 1, n
       A(i,j) = i + j
    end do
  end do
  !$OMP END SIMD
end subroutine
