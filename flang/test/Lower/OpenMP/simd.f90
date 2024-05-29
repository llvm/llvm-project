! Tests for 2.9.3.1 Simd

! The "if" clause was added to the "simd" directive in OpenMP 5.0.
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: bbc -hlfir -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsimd()
subroutine simd
  integer :: i
  !$OMP SIMD
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = arith.constant 9 : i32
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.simd {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i=1, 9
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_if_clause
subroutine simd_with_if_clause(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_if_clauseEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  !$OMP SIMD IF( n .GE. threshold )
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: %[[COND:.*]] = arith.cmpi sge
  ! CHECK: omp.simd if(%[[COND:.*]]) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_simdlen_clause
subroutine simd_with_simdlen_clause(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_simdlen_clauseEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  !$OMP SIMD SIMDLEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd simdlen(2) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_simdlen_clause_from_param
subroutine simd_with_simdlen_clause_from_param(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_simdlen_clause_from_paramEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  integer, parameter :: simdlen = 2;
  !$OMP SIMD SIMDLEN(simdlen)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd simdlen(2) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_simdlen_clause_from_expr_from_param
subroutine simd_with_simdlen_clause_from_expr_from_param(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_simdlen_clause_from_expr_from_paramEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  integer, parameter :: simdlen = 2;
  !$OMP SIMD SIMDLEN(simdlen*2 + 2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd simdlen(6) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_safelen_clause
subroutine simd_with_safelen_clause(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_safelen_clauseEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  !$OMP SIMD SAFELEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd safelen(2) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_safelen_clause_from_expr_from_param
subroutine simd_with_safelen_clause_from_expr_from_param(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_safelen_clause_from_expr_from_paramEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  integer, parameter :: safelen = 2;
  !$OMP SIMD SAFELEN(safelen*2 + 2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd safelen(6) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_simdlen_safelen_clause
subroutine simd_with_simdlen_safelen_clause(n, threshold)
  ! CHECK: %[[ARG_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFsimd_with_simdlen_safelen_clauseEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n, threshold
  !$OMP SIMD SIMDLEN(1) SAFELEN(2)
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd simdlen(1) safelen(2) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  do i = 1, n
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[LD]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimd_with_collapse_clause
subroutine simd_with_collapse_clause(n)
  integer :: i, j, n
  integer :: A(n,n)
  ! CHECK: %[[LOWER_I:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UPPER_I:.*]] = fir.load %[[PARAM_ARG:.*]] : !fir.ref<i32>
  ! CHECK: %[[STEP_I:.*]] = arith.constant 1 : i32
  ! CHECK: %[[LOWER_J:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UPPER_J:.*]] = fir.load %[[PARAM_ARG:.*]] : !fir.ref<i32>
  ! CHECK: %[[STEP_J:.*]] = arith.constant 1 : i32
  ! CHECK: omp.simd {
  ! CHECK-NEXT: omp.loop_nest (%[[ARG_0:.*]], %[[ARG_1:.*]]) : i32 = (
  ! CHECK-SAME:                %[[LOWER_I]], %[[LOWER_J]]) to (
  ! CHECK-SAME:                %[[UPPER_I]], %[[UPPER_J]]) inclusive step (
  ! CHECK-SAME:                %[[STEP_I]], %[[STEP_J]]) {
  !$OMP SIMD COLLAPSE(2)
  do i = 1, n
    do j = 1, n
       A(i,j) = i + j
    end do
  end do
  !$OMP END SIMD
end subroutine
