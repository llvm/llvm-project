! This test checks lowering of OpenMP linear clause with implicit modifiers.

! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,OPENMP52
! RUN: %flang_fc1 -fopenmp -fopenmp-version=45 -emit-hlfir %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,OPENMP45

!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_linearEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFsimple_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[const:.*]] = arith.constant 1 : i32
subroutine simple_linear
    implicit none
    integer :: x, y, i
    !OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = %[[const]] : i32)) {{.*}}
    !OPENMP45: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = %[[const]] : i32) {{.*}}
    !$omp do linear(x)
    do i = 1, 10
        y = x + 2
    end do
    !$omp end do
    !CHECK: } {linear_var_types = [i32]}
end subroutine

subroutine linear_step
!CHECK: %[[X_alloca:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlinear_stepEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_alloca]] {uniq_name = "_QFlinear_stepEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    implicit none
    integer :: x, y, i
    !CHECK: %[[const:.*]] = arith.constant 4 : i32
    !OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = %[[const]] : i32)) {{.*}}
    !OPENMP45: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = %[[const]] : i32) {{.*}}
    !$omp do linear(x:4)
    do i = 1, 10
        y = x + 2
    end do
    !$omp end do
    !CHECK: } {linear_var_types = [i32]}
end subroutine

subroutine do_simd_linear
!CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFdo_simd_linearEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFdo_simd_linearEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 1 : i32
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: %[[IV_STEP:.*]] = arith.constant 1 : i32
!OPENMP52: omp.wsloop linear(val(%[[X]]#0 : !fir.ref<i32> = %[[CONST]] : i32)) {
!OPENMP45: omp.wsloop linear(%[[X]]#0 : !fir.ref<i32> = %[[CONST]] : i32) {
!OPENMP52: omp.simd linear(val(%[[I]]#0 : !fir.ref<i32> = %[[IV_STEP]] : i32)) private({{.*}}) {
!OPENMP45: omp.simd linear(%[[I]]#0 : !fir.ref<i32> = %[[IV_STEP]] : i32) private({{.*}}) {
    integer :: x
    !$omp do simd linear(x:1)
    do i = 1, 10
    end do
    !$omp end do simd
!CHECK: } {linear_var_types = [i32], omp.composite}
!CHECK: } {linear_var_types = [i32], omp.composite}
end subroutine do_simd_linear
