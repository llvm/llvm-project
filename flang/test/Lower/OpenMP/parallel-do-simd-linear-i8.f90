! Test lowering of OpenMP parallel do simd with linear clause on INTEGER(8) variable
! This is a regression test for issue #173332
! At MLIR level, the step type may differ from the variable type - this is legal and handled during translation

! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_linear_i8
subroutine test_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK-DAG: %[[J_alloca:.*]] = fir.alloca i64 {bindc_name = "j"
    ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_alloca]]
    ! CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.wsloop
    ! CHECK: omp.simd linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp parallel do simd linear(j)
    do i = 1,100,1
    end do
    !$omp end parallel do simd
end subroutine

! CHECK-LABEL: func.func @_QPtest_linear_i8_with_step
subroutine test_linear_i8_with_step
    implicit none
    integer(8) :: i, j
    
    ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_linear_i8_with_stepEj"}
    ! CHECK: omp.wsloop
    ! CHECK: omp.simd linear(%[[J]]#0 = {{.*}} : !fir.ref<i64>)
    !$omp parallel do simd linear(j:2)
    do i = 1,100,1
    end do
    !$omp end parallel do simd
end subroutine

! CHECK-LABEL: func.func @_QPtest_simd_linear_i8
subroutine test_simd_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_simd_linear_i8Ej"}
    ! CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.simd linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp simd linear(j)
    do i = 1,100,1
    end do
    !$omp end simd
end subroutine

! CHECK-LABEL: func.func @_QPtest_do_linear_i8
subroutine test_do_linear_i8
    implicit none
    integer(8) :: i, j
    
    ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_do_linear_i8Ej"}
    ! CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.wsloop linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp do linear(j)
    do i = 1,100,1
    end do
    !$omp end do
end subroutine

! Test with multiple INTEGER(8) linear variables
! CHECK-LABEL: func.func @_QPtest_multiple_i8
subroutine test_multiple_i8
    implicit none
    integer(8) :: i, j, k
    
    ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_multiple_i8Ej"}
    ! CHECK-DAG: %[[K:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_multiple_i8Ek"}
    ! CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : i32
    ! CHECK: omp.simd linear(%[[J]]#0 = %[[STEP]] : !fir.ref<i64>, %[[K]]#0 = %[[STEP]] : !fir.ref<i64>)
    !$omp simd linear(j,k)
    do i = 1,100,1
    end do
    !$omp end simd
end subroutine
