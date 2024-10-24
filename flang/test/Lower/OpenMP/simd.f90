! Tests for 2.9.3.1 Simd

! The "if" clause was added to the "simd" directive in OpenMP 5.0.
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: bbc -hlfir -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK: omp.declare_reduction @[[REDUCER:.*]] : i32

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
  ! CHECK: %[[COND:.*]] = arith.cmpi sge
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = fir.load %[[ARG_N]]#0
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
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
  ! CHECK: omp.simd safelen(2) simdlen(1) {
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


!CHECK: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!CHECK-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!CHECK-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG_A]] dummy_scope %0
!CHECK-SAME: {uniq_name = "_QFsimdloop_aligned_cptrEa"} :
!CHECK-SAME: (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) ->
!CHECK-SAME: (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>,
!CHECK-SAME: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
subroutine simdloop_aligned_cptr( A)
  use iso_c_binding
  integer :: i
  type (c_ptr) :: A
!CHECK: omp.simd aligned(%[[A_DECL]]#1 : !fir.ref
!CHECK-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
!CHECK-SAME: -> 256 : i64)
  !$OMP SIMD ALIGNED(A:256)
  do i = 1, 10
    call c_test_call(A)
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func @_QPsimdloop_aligned_allocatable
subroutine simdloop_aligned_allocatable()
  integer :: i
  integer, allocatable :: A(:)
  allocate(A(10))
!CHECK: %[[A_PTR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a",
!CHECK-SAME: uniq_name = "_QFsimdloop_aligned_allocatableEa"}
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A_PTR]] {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME: uniq_name = "_QFsimdloop_aligned_allocatableEa"} :
!CHECK-SAME: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK: omp.simd aligned(%[[A_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> -> 256 : i64)
  !$OMP SIMD ALIGNED(A:256)
  do i = 1, 10
    A(i) = i
  end do
end subroutine

!CHECK-LABEL: func @_QPsimd_with_nontemporal_clause
subroutine simd_with_nontemporal_clause(n)
  !CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: i, n
  integer :: A, B, C
  !CHECK: %[[LB:.*]] = arith.constant 1 : i32
  !CHECK: %[[UB:.*]] = fir.load %{{.*}}#0 : !fir.ref<i32>
  !CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  !CHECK: omp.simd nontemporal(%[[A_DECL]]#1, %[[C_DECL]]#1 : !fir.ref<i32>, !fir.ref<i32>) {
  !CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  !$OMP SIMD NONTEMPORAL(A, C)
  do i = 1, n
    C = A + B
  end do
  !$OMP END SIMD
end subroutine

!CHECK-LABEL: func.func @_QPlastprivate_with_simd() {
subroutine lastprivate_with_simd

!CHECK: %[[VAR_SUM:.*]] = fir.alloca f32 {bindc_name = "sum", uniq_name = "_QFlastprivate_with_simdEsum"}
!CHECK: %[[VAR_SUM_DECLARE:.*]]:2 = hlfir.declare %[[VAR_SUM]] {{.*}}
!CHECK: %[[VAR_SUM_PINNED:.*]] = fir.alloca f32 {bindc_name = "sum", pinned, uniq_name = "_QFlastprivate_with_simdEsum"}
!CHECK: %[[VAR_SUM_PINNED_DECLARE:.*]]:2 = hlfir.declare %[[VAR_SUM_PINNED]] {{.*}}

  implicit none
  integer :: i
  real :: sum

  
!CHECK: omp.simd {
!CHECK: omp.loop_nest (%[[ARG:.*]]) : i32 = ({{.*}} to ({{.*}}) inclusive step ({{.*}}) {
!CHECK: %[[ADD_RESULT:.*]] = arith.addi {{.*}}
!CHECK: %[[ADD_RESULT_CONVERT:.*]] = fir.convert %[[ADD_RESULT]] : (i32) -> f32
!CHECK: hlfir.assign %[[ADD_RESULT_CONVERT]] to %[[VAR_SUM_PINNED_DECLARE]]#0 : f32, !fir.ref<f32>
!CHECK: %[[SELECT_RESULT:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : i1
!CHECK: fir.if %[[SELECT_RESULT]] {
!CHECK: %[[LOADED_SUM:.*]] = fir.load %[[VAR_SUM_PINNED_DECLARE]]#0 : !fir.ref<f32>
!CHECK: hlfir.assign %[[LOADED_SUM]] to %[[VAR_SUM_DECLARE]]#0 : f32, !fir.ref<f32>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: }
  !$omp simd lastprivate(sum)
  do i = 1, 100
    sum = i + 1
  end do
end subroutine

!CHECK-LABEL: func @_QPsimd_with_reduction_clause()
subroutine simd_with_reduction_clause
  integer :: i, x
  x = 0
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = arith.constant 9 : i32
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.simd reduction(@[[REDUCER]] %[[X:.*]]#0 -> %[[X_RED:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  !$omp simd reduction(+:x)
  do i=1, 9
    ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_RED]] {uniq_name = "_QFsimd_with_reduction_clauseEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK: fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
    ! CHECK: %[[X_LD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
    ! CHECK: %[[I_LD:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
    ! CHECK: %[[SUM:.*]] = arith.addi %[[X_LD]], %[[I_LD]] : i32
    ! CHECK: hlfir.assign %[[SUM]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
    x = x+i
  end do
  !$OMP end simd
end subroutine
