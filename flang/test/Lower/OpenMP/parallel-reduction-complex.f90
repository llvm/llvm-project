! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME: @[[RED_NAME:.*]] : !fir.complex<8> init {
!CHECK: ^bb0(%{{.*}}: !fir.complex<8>):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:  %[[C0_2:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:  %[[UNDEF:.*]] = fir.undefined !fir.complex<8>
!CHECK:  %[[RES_1:.*]] = fir.insert_value %[[UNDEF]], %[[C0_1]], [0 : index]
!CHECK:  %[[RES_2:.*]] = fir.insert_value %[[RES_1]], %[[C0_2]], [1 : index]
!CHECK:  omp.yield(%[[RES_2]] : !fir.complex<8>)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: !fir.complex<8>, %[[ARG1:.*]]: !fir.complex<8>):
!CHECK:  %[[RES:.*]] = fir.addc %[[ARG0]], %[[ARG1]] {{.*}}: !fir.complex<8>
!CHECK:  omp.yield(%[[RES]] : !fir.complex<8>)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_complex_add
!CHECK:  %[[CREF:.*]] = fir.alloca !fir.complex<8> {bindc_name = "c", {{.*}}}
!CHECK:  %[[C_DECL:.*]]:2 = hlfir.declare %[[CREF]] {uniq_name = "_QFsimple_complex_addEc"} : (!fir.ref<!fir.complex<8>>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>)
!CHECK:  %[[C_START_RE:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:  %[[C_START_IM:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:  %[[UNDEF_1:.*]] = fir.undefined !fir.complex<8>
!CHECK:  %[[VAL_1:.*]] = fir.insert_value %[[UNDEF_1]], %[[C_START_RE]], [0 : index]
!CHECK:  %[[VAL_2:.*]] = fir.insert_value %[[VAL_1]], %[[C_START_IM]], [1 : index]
!CHECK:  hlfir.assign %[[VAL_2]] to %[[C_DECL]]#0 : !fir.complex<8>, !fir.ref<!fir.complex<8>>
!CHECK:  omp.parallel reduction(@[[RED_NAME]] %[[C_DECL]]#0 -> %[[PRV:.+]] : !fir.ref<!fir.complex<8>>) {
!CHECK:    %[[P_DECL:.+]]:2 = hlfir.declare %[[PRV]] {{.*}} : (!fir.ref<!fir.complex<8>>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>)
!CHECK:    %[[LPRV:.+]] = fir.load %[[P_DECL]]#0 : !fir.ref<!fir.complex<8>>
!CHECK:    %[[C_INCR_RE:.*]] = arith.constant 1.000000e+00 : f64
!CHECK:    %[[C_INCR_IM:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:    %[[UNDEF_2:.*]] = fir.undefined !fir.complex<8>
!CHECK:    %[[INCR_1:.*]] = fir.insert_value %[[UNDEF_2]], %[[C_INCR_RE]], [0 : index]
!CHECK:    %[[INCR_2:.*]] = fir.insert_value %[[INCR_1]], %[[C_INCR_IM]], [1 : index]
!CHECK:    %[[RES:.+]] = fir.addc %[[LPRV]], %[[INCR_2]] {{.*}} : !fir.complex<8>
!CHECK:    hlfir.assign %[[RES]] to %[[P_DECL]]#0 : !fir.complex<8>, !fir.ref<!fir.complex<8>>
!CHECK:    omp.terminator
!CHECK:  }
!CHECK: return
subroutine simple_complex_add
    complex(8) :: c
    c = 0

    !$omp parallel reduction(+:c)
    c = c + 1
    !$omp end parallel

    print *, c
end subroutine
