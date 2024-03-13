! RUN: bbc -emit-fir -hlfir=false -fopenmp --force-byref-reduction -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp -mmlir --force-byref-reduction -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_F32_NAME:.*]] : !fir.ref<f32>
!CHECK-SAME: init {
!CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  %[[REF:.*]] = fir.alloca f32
!CHECKL  fir.store [[%C0_1]] to %[[REF]] : !fir.ref<f32>
!CHECK:  omp.yield(%[[REF]] : !fir.ref<f32>)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
!CHECK:  %[[LD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
!CHECK:  %[[LD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
!CHECK:  %[[RES:.*]] = arith.addf %[[LD0]], %[[LD1]] {{.*}}: f32
!CHECK:  fir.store %[[RES]] to %[[ARG0]] : !fir.ref<f32>
!CHECK:  omp.yield(%[[ARG0]] : !fir.ref<f32>)
!CHECK: }

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_I32_NAME:.*]] : !fir.ref<i32>
!CHECK-SAME: init {
!CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:  %[[REF:.*]] = fir.alloca i32
!CHECKL  fir.store [[%C0_1]] to %[[REF]] : !fir.ref<i32>
!CHECK:  omp.yield(%[[REF]] : !fir.ref<i32>)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
!CHECK:  %[[LD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
!CHECK:  %[[LD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
!CHECK:  %[[RES:.*]] = arith.addi %[[LD0]], %[[LD1]] : i32
!CHECK:  fir.store %[[RES]] to %[[ARG0]] : !fir.ref<i32>
!CHECK:  omp.yield(%[[ARG0]] : !fir.ref<i32>)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_int_add
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_int_addEi"}
!CHECK:  %[[I_START:.*]] = arith.constant 0 : i32
!CHECK:  fir.store %[[I_START]] to %[[IREF]] : !fir.ref<i32>
!CHECK:  omp.parallel byref reduction(@[[RED_I32_NAME]] %[[IREF]] -> %[[PRV:.+]] : !fir.ref<i32>) {
!CHECK:    %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<i32>
!CHECK:    %[[I_INCR:.+]] = arith.constant 1 : i32
!CHECK:    %[[RES:.+]] = arith.addi %[[LPRV]], %[[I_INCR]]
!CHECK:    fir.store %[[RES]] to %[[PRV]] : !fir.ref<i32>
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
!CHECK:  omp.parallel byref reduction(@[[RED_F32_NAME]] %[[RREF]] -> %[[PRV:.+]] : !fir.ref<f32>) {
!CHECK:    %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<f32>
!CHECK:    %[[R_INCR:.+]] = arith.constant 1.500000e+00 : f32
!CHECK:    %[[RES]] = arith.addf %[[LPRV]], %[[R_INCR]] {{.*}} : f32
!CHECK:    fir.store %[[RES]] to %[[PRV]] : !fir.ref<f32>
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
!CHECK:  omp.parallel byref reduction(@[[RED_I32_NAME]] %[[IREF]] -> %[[PRV0:.+]] : !fir.ref<i32>, @[[RED_F32_NAME]] %[[RREF]] -> %[[PRV1:.+]] : !fir.ref<f32>) {
!CHECK:    %[[R_INCR:.*]] = arith.constant 1.500000e+00 : f32
!CHECK:    %[[LPRV1:.+]] = fir.load %[[PRV1]] : !fir.ref<f32>
!CHECK:    %[[RES1:.+]] = arith.addf %[[R_INCR]], %[[LPRV1]] {{.*}} : f32
!CHECK:    fir.store %[[RES1]] to %[[PRV1]]
!CHECK:    %[[LPRV0:.+]] = fir.load %[[PRV0]] : !fir.ref<i32>
!CHECK:    %[[I_INCR:.*]] = arith.constant 3 : i32
!CHECK:    %[[RES0:.+]] = arith.addi %[[LPRV0]], %[[I_INCR]]
!CHECK:    fir.store %[[RES0]] to %[[PRV0]]
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
