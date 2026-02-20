! This test checks lowering of OpenMP SIMD Directive
! with linear clause

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=60 %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s --check-prefix=IMPLICIT

!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_linearEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFsimple_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[const:.*]] = arith.constant 1 : i32

!IMPLICIT: %[[I_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_linearEi"}
!IMPLICIT: %[[I:.*]]:2 = hlfir.declare %[[I_ALLOCA]] {{.*}} 
!IMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_linearEx"}
!IMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFsimple_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!IMPLICIT: %[[const:.*]] = arith.constant 1 : i32
subroutine simple_linear
    implicit none
    integer :: x, y, i
    !CHECK: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>) {{.*}}

    !IMPLICIT: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>, %[[I]]#0 = %{{.*}} : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x)
    do i = 1, 10
    end do
    !CHECK: } {linear_var_types = [i32]}
    !IMPLICIT: } {linear_var_types = [i32, i32]}
end subroutine


!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_stepEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_stepEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!IMPLICIT: %[[I_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlinear_stepEi"}
!IMPLICIT: %[[I:.*]]:2 = hlfir.declare %[[I_ALLOCA]] {{.*}} 
!IMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_stepEx"}
!IMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_stepEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!IMPLICIT: %[[const:.*]] = arith.constant 4 : i32
subroutine linear_step
    implicit none
    integer :: x, y, i
    !CHECK: %[[const:.*]] = arith.constant 4 : i32
    !CHECK: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>) {{.*}}

    !IMPLICIT: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>, %[[I]]#0 = %{{.*}} : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x:4)
    do i = 1, 10
    end do
    !CHECK: } {linear_var_types = [i32]}
    !IMPLICIT: } {linear_var_types = [i32, i32]}
end subroutine

!CHECK: %[[A_alloca:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFlinear_exprEa"}
!CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_alloca]] {uniq_name = "_QFlinear_exprEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_exprEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_exprEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!IMPLICIT: %[[A_alloca:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFlinear_exprEa"}
!IMPLICIT: %[[A:.*]]:2 = hlfir.declare %[[A_alloca]] {uniq_name = "_QFlinear_exprEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!IMPLICIT: %[[I_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlinear_exprEi"}
!IMPLICIT: %[[I:.*]]:2 = hlfir.declare %[[I_ALLOCA]] {uniq_name = "_QFlinear_exprEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!IMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_exprEx"}
!IMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_exprEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine linear_expr
    implicit none
    integer :: x, y, i, a
    !CHECK: %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
    !CHECK: %[[const:.*]] = arith.constant 4 : i32
    !CHECK: %[[LINEAR_EXPR:.*]] = arith.addi %[[LOAD_A]], %[[const]] : i32

    !IMPLICIT: %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
    !IMPLICIT: %[[const:.*]] = arith.constant 4 : i32
    !IMPLICIT: %[[LINEAR_EXPR:.*]] = arith.addi %[[LOAD_A]], %[[const]] : i32

    !CHECK: omp.simd linear(%[[X]]#0 = %[[LINEAR_EXPR]] : !fir.ref<i32>) {{.*}}

    !IMPLICIT: omp.simd linear(%[[X]]#0 = %[[LINEAR_EXPR]] : !fir.ref<i32>, %[[I]]#0 = {{.*}} : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x:a+4)
    do i = 1, 10
    end do
    !CHECK: } {linear_var_types = [i32]}
    !IMPLICIT: } {linear_var_types = [i32, i32]}
end subroutine

subroutine simple_linear_i8
  implicit none
  integer(kind=8) :: x, y, i

  ! CHECK-LABEL: func.func @_QPsimple_linear_i8

  ! CHECK-DAG: %[[X_ALLOC:.*]] = fir.alloca i64 {bindc_name = "x"
  ! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32

  ! CHECK: omp.simd linear(%[[X_DECL:.*]]#0 = %[[C1_I32]] : !fir.ref<i64>) {{.*}}

  !$omp simd linear(x)
  do i = 1_8, 10_8
  end do

  ! CHECK: } {linear_var_types = [i64]}
end subroutine
