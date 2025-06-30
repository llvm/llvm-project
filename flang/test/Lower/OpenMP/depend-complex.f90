! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s | FileCheck %s

subroutine depend_complex(z)
! CHECK-LABEL:   func.func @_QPdepend_complex(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "z"}) {
  complex :: z
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFdepend_complexEz"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  !$omp task depend(in:z%re)
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0  real : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
! CHECK:           omp.task depend(taskdependin -> %[[VAL_2]] : !fir.ref<f32>) {
! CHECK:             omp.terminator
! CHECK:           }
  !$omp end task
  !$omp task depend(in:z%im)
! CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#0  imag : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
! CHECK:           omp.task depend(taskdependin -> %[[VAL_3]] : !fir.ref<f32>) {
! CHECK:             omp.terminator
! CHECK:           }
  !$omp end task
end subroutine

