! This test checks lowering of OpenMP SIMD with linear clause

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - 2>&1 | FileCheck %s --check-prefix=NOIMPLICIT

!CHECK: %[[IV_alloca:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_linearEi"}
!CHECK: %[[IV:.*]]:2 = hlfir.declare %[[IV_alloca]] {uniq_name = "_QFsimple_linearEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_linearEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFsimple_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[const:.*]] = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %[[IV_step:.*]] = arith.constant 1 : i32

!NOIMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_linearEx"}
!NOIMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFsimple_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!NOIMPLICIT: %[[const:.*]] = arith.constant 1 : i32
subroutine simple_linear
    implicit none
    integer :: x, y, i
    !CHECK: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>, %[[IV]]#0 = %[[IV_step]] : !fir.ref<i32>) {{.*}}
    !NOIMPLICIT: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x)
    !CHECK: %[[LOAD:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
    !CHECK: %[[const:.*]] = arith.constant 2 : i32
    !CHECK: %[[RESULT:.*]] = arith.addi %[[LOAD]], %[[const]] : i32
    do i = 1, 10
        y = x + 2
    end do
end subroutine

!CHECK: %[[IV_alloca:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlinear_stepEi"}
!CHECK: %[[IV:.*]]:2 = hlfir.declare %[[IV_alloca]] {uniq_name = "_QFlinear_stepEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_stepEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_stepEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!NOIMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_stepEx"}
!NOIMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_stepEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!NOIMPLICIT: %[[const:.*]] = arith.constant 4 : i32
subroutine linear_step
    implicit none
    integer :: x, y, i
    !CHECK: %[[const:.*]] = arith.constant 4 : i32
    !CHECK: %{{.*}} = arith.constant 1 : i32
    !CHECK: %[[IV_step:.*]] = arith.constant 1 : i32
    !CHECK: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>, %[[IV]]#0 = %[[IV_step]] : !fir.ref<i32>) {{.*}}

    !NOIMPLICIT: omp.simd linear(%[[X]]#0 = %[[const]] : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x:4)
    !CHECK: %[[LOAD:.*]] = fir.load %[[X]]#0 : !fir.ref<i32>
    !CHECK: %[[const:.*]] = arith.constant 2 : i32
    !CHECK: %[[RESULT:.*]] = arith.addi %[[LOAD]], %[[const]] : i32   
    do i = 1, 10
        y = x + 2
    end do
end subroutine

!CHECK: %[[A_alloca:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFlinear_exprEa"}
!CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_alloca]] {uniq_name = "_QFlinear_exprEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[IV_alloca:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlinear_exprEi"}
!CHECK: %[[IV:.*]]:2 = hlfir.declare %[[IV_alloca]] {uniq_name = "_QFlinear_exprEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_exprEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_exprEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!NOIMPLICIT: %[[A_alloca:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFlinear_exprEa"}
!NOIMPLICIT: %[[A:.*]]:2 = hlfir.declare %[[A_alloca]] {uniq_name = "_QFlinear_exprEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!NOIMPLICIT: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_exprEx"}
!NOIMPLICIT: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_exprEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!NOIMPLICIT: %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
!NOIMPLICIT: %[[const:.*]] = arith.constant 4 : i32
!NOIMPLICIT: %[[LINEAR_EXPR:.*]] = arith.addi %[[LOAD_A]], %[[const]] : i32
subroutine linear_expr
    implicit none
    integer :: x, y, i, a
    !CHECK: %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
    !CHECK: %[[const:.*]] = arith.constant 4 : i32
    !CHECK: %[[LINEAR_EXPR:.*]] = arith.addi %[[LOAD_A]], %[[const]] : i32
    !CHECK: %{{.*}} = arith.constant 1 : i32
    !CHECK: %[[IV_step:.*]] = arith.constant 1 : i32
    !CHECK: omp.simd linear(%[[X]]#0 = %[[LINEAR_EXPR]] : !fir.ref<i32>, %[[IV]]#0 = %[[IV_step]] : !fir.ref<i32>) {{.*}}

    !NOIMPLICIT: omp.simd linear(%[[X]]#0 = %[[LINEAR_EXPR]] : !fir.ref<i32>) {{.*}}
    !$omp simd linear(x:a+4)
    do i = 1, 10
        y = x + 2
    end do
end subroutine
