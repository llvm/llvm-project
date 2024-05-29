! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME: @[[RED_F32_NAME:.*]] : f32 init {
!CHECK: ^bb0(%{{.*}}: f32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  omp.yield(%[[C0_1]] : f32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
!CHECK:  %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}}: f32
!CHECK:  omp.yield(%[[RES]] : f32)
!CHECK: }

!CHECK-LABEL: omp.declare_reduction
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
!CHECK:  %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {uniq_name = "_QFsimple_int_addEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[I_START:.*]] = arith.constant 0 : i32
!CHECK:  hlfir.assign %[[I_START]] to %[[I_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:  omp.parallel reduction(@[[RED_I32_NAME]] %[[I_DECL]]#0 -> %[[PRV:.+]] : !fir.ref<i32>) {
!CHECK:    %[[P_DECL:.+]]:2 = hlfir.declare %[[PRV]] {{.*}} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[LPRV:.+]] = fir.load %[[P_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[I_INCR:.*]] = arith.constant 1 : i32
!CHECK:    %[[RES:.+]] = arith.addi %[[LPRV]], %[[I_INCR]] : i32
!CHECK:    hlfir.assign %[[RES]] to %[[P_DECL]]#0 : i32, !fir.ref<i32>
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
!CHECK:  %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {uniq_name = "_QFsimple_real_addEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[R_START:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  hlfir.assign %[[R_START]] to %[[R_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:  omp.parallel reduction(@[[RED_F32_NAME]] %[[R_DECL]]#0 -> %[[PRV:.+]] : !fir.ref<f32>) {
!CHECK:    %[[P_DECL:.+]]:2 = hlfir.declare %[[PRV]] {{.*}} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:    %[[LPRV:.+]] = fir.load %[[P_DECL]]#0 : !fir.ref<f32>
!CHECK:    %[[R_INCR:.*]] = arith.constant 1.500000e+00 : f32
!CHECK:    %[[RES:.+]] = arith.addf %[[LPRV]], %[[R_INCR]] {{.*}} : f32
!CHECK:    hlfir.assign %[[RES]] to %[[P_DECL]]#0 : f32, !fir.ref<f32>
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
!CHECK:  %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {uniq_name = "_QFint_real_addEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[RREF:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFint_real_addEr"}
!CHECK:  %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {uniq_name = "_QFint_real_addEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[R_START:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  hlfir.assign %[[R_START]] to %[[R_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:  %[[I_START:.*]] = arith.constant 0 : i32
!CHECK:  hlfir.assign %[[I_START]] to %[[I_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:  omp.parallel reduction(@[[RED_I32_NAME]] %[[I_DECL]]#0 -> %[[IPRV:.+]] : !fir.ref<i32>, @[[RED_F32_NAME]] %[[R_DECL]]#0 -> %[[RPRV:.+]] : !fir.ref<f32>) {
!CHECK:    %[[IP_DECL:.+]]:2 = hlfir.declare %[[IPRV]] {{.*}} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[RP_DECL:.+]]:2 = hlfir.declare %[[RPRV]] {{.*}} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:    %[[R_INCR:.*]] = arith.constant 1.500000e+00 : f32
!CHECK:    %[[R_LPRV:.+]] = fir.load %[[RP_DECL]]#0 : !fir.ref<f32>
!CHECK:    %[[RES1:.+]] = arith.addf %[[R_INCR]], %[[R_LPRV]] {{.*}} : f32
!CHECK:    hlfir.assign %[[RES1]] to %[[RP_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:    %[[I_LPRV:.+]] = fir.load %[[IP_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[I_INCR:.*]] = arith.constant 3 : i32
!CHECK:    %[[RES0:.+]] = arith.addi %[[I_LPRV]], %[[I_INCR]] : i32
!CHECK:    hlfir.assign %[[RES0]] to %[[IP_DECL]]#0 : i32, !fir.ref<i32>
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
