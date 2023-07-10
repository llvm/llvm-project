! RUN: bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_F32_NAME:.*]] : f32 init {
!CHECK: ^bb0(%{{.*}}: f32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  omp.yield(%[[C0_1]] : f32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
!CHECK:  %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}}: f32
!CHECK:  omp.yield(%[[RES]] : f32)
!CHECK: }

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_I32_NAME:.*]] : i32 init {
!CHECK: ^bb0(%{{.*}}: i32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:  omp.yield(%[[C0_1]] : i32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:  %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:  omp.yield(%[[RES]] : i32)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_int_add
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_int_addEi"}
!CHECK:  %[[I_START:.*]] = arith.constant 0 : i32
!CHECK:  fir.store %[[I_START]] to %[[IREF]] : !fir.ref<i32>
!CHECK:  omp.parallel   reduction(@[[RED_I32_NAME]] -> %[[IREF]] : !fir.ref<i32>) {
!CHECK:    %[[I_INCR:.*]] = arith.constant 1 : i32
!CHECK:    omp.reduction %[[I_INCR]], %[[IREF]] : i32, !fir.ref<i32>
!CHECK:    omp.terminator
!CHECK:  }
!CHECK: return
subroutine simple_int_add
    integer :: i
    i = 0

    !$omp parallel reduction(+:i)
    i = i + 1
    !$omp end parallel

    print *, i
end subroutine

!CHECK-LABEL: func.func @_QPsimple_real_add
!CHECK:  %[[RREF:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFsimple_real_addEr"}
!CHECK:  %[[R_START:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  fir.store %[[R_START]] to %[[RREF]] : !fir.ref<f32>
!CHECK:  omp.parallel   reduction(@[[RED_F32_NAME]] -> %[[RREF]] : !fir.ref<f32>) {
!CHECK:    %[[R_INCR:.*]] = arith.constant 1.500000e+00 : f32
!CHECK:    omp.reduction %[[R_INCR]], %[[RREF]] : f32, !fir.ref<f32>
!CHECK:    omp.terminator
!CHECK:  }
!CHECK: return
subroutine simple_real_add
    real :: r
    r = 0.0

    !$omp parallel reduction(+:r)
    r = r + 1.5
    !$omp end parallel

    print *, r
end subroutine

!CHECK-LABEL: func.func @_QPint_real_add
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFint_real_addEi"}
!CHECK:  %[[RREF:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFint_real_addEr"}
!CHECK:  %[[R_START:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  fir.store %[[R_START]] to %[[RREF]] : !fir.ref<f32>
!CHECK:  %[[I_START:.*]] = arith.constant 0 : i32
!CHECK:  fir.store %[[I_START]] to %[[IREF]] : !fir.ref<i32>
!CHECK:  omp.parallel   reduction(@[[RED_I32_NAME]] -> %[[IREF]] : !fir.ref<i32>, @[[RED_F32_NAME]] -> %[[RREF]] : !fir.ref<f32>) {
!CHECK:    %[[R_INCR:.*]] = arith.constant 1.500000e+00 : f32
!CHECK:    omp.reduction %[[R_INCR]], %[[RREF]] : f32, !fir.ref<f32>
!CHECK:    %[[I_INCR:.*]] = arith.constant 3 : i32
!CHECK:    omp.reduction %[[I_INCR]], %[[IREF]] : i32, !fir.ref<i32>
!CHECK:    omp.terminator
!CHECK:  }
!CHECK: return
subroutine int_real_add
    real :: r
    integer :: i

    r = 0.0
    i = 0

    !$omp parallel reduction(+:i,r)
    r = 1.5 + r
    i = i + 3
    !$omp end parallel

    print *, r
    print *, i
end subroutine
