! Test lowering of OpenMP parallel do simd with linear clause on INTEGER(8) variable
! This is a regression test for issue #173332
! At MLIR level, the step type may differ from the variable type - this is legal and handled during translation

! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: func @_QPtest_linear_i8
subroutine test_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j", uniq_name = "_QFtest_linear_i8Ej"}
    ! CHECK: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]] {uniq_name = "_QFtest_linear_i8Ej"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.parallel
    ! CHECK: omp.wsloop
    ! CHECK-SAME: linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp parallel do simd linear(j)
    do i = 1,100,1
    end do
    !$omp end parallel do simd
    ! CHECK: } {linear_var_types = [i64]}
end subroutine

! CHECK-LABEL: func @_QPtest_linear_i8_with_step
subroutine test_linear_i8_with_step
    implicit none
    integer(8) :: i, j
    
    ! CHECK: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j", uniq_name = "_QFtest_linear_i8_with_stepEj"}
    ! CHECK: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]] {uniq_name = "_QFtest_linear_i8_with_stepEj"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! Step may be i32 at MLIR level - translation will handle type normalization
    ! CHECK: omp.parallel
    ! CHECK: omp.wsloop
    ! CHECK-SAME: linear(%[[J]]#0 = {{.*}} : !fir.ref<i64>)
    !$omp parallel do simd linear(j:2)
    do i = 1,100,1
    end do
    !$omp end parallel do simd
    ! CHECK: } {linear_var_types = [i64]}
end subroutine

! CHECK-LABEL: func @_QPtest_simd_linear_i8
subroutine test_simd_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j", uniq_name = "_QFtest_simd_linear_i8Ej"}
    ! CHECK: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]] {uniq_name = "_QFtest_simd_linear_i8Ej"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.simd linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp simd linear(j)
    do i = 1,100,1
    end do
    !$omp end simd
    ! CHECK: } {linear_var_types = [i64]}
end subroutine

! CHECK-LABEL: func @_QPtest_do_linear_i8
subroutine test_do_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j", uniq_name = "_QFtest_do_linear_i8Ej"}
    ! CHECK: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]] {uniq_name = "_QFtest_do_linear_i8Ej"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.wsloop linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp do linear(j)
    do i = 1,100,1
    end do
    !$omp end do
    ! CHECK: } {linear_var_types = [i64]}
end subroutine

! Test with multiple INTEGER(8) linear variables
! CHECK-LABEL: func @_QPtest_multiple_i8
subroutine test_multiple_i8
    implicit none
    integer(8) :: i, j, k
    
    ! CHECK: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j"
    ! CHECK: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]]
    ! CHECK: %[[K_alloca:.*]] = fir.alloca i64 {bindc_name = "k"
    ! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_alloca]]
    ! CHECK: %[[STEP_J:.*]] = arith.constant 1 : i64
    ! CHECK: %[[STEP_K:.*]] = arith.constant 1 : i64
    ! CHECK: omp.simd linear(%[[J]]#0 = %[[STEP_J]] : !fir.ref<i64>, %[[K]]#0 = %[[STEP_K]] : !fir.ref<i64>)
    !$omp simd linear(j,k)
    do i = 1,100,1
    end do
    !$omp end simd
    ! CHECK: } {linear_var_types = [i64, i64]}
end subroutine
