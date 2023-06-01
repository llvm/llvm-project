! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.reduction.recipe @reduction_add_f32 : f32 reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: f32):
! CHECK:   %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   acc.yield %[[INIT]] : f32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
! CHECK:   %[[COMBINED:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}} : f32
! CHECK:   acc.yield %[[COMBINED]] : f32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_i32 : i32 reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: i32):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   acc.yield %[[INIT]] : i32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
! CHECK:   %[[COMBINED:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
! CHECK:   acc.yield %[[COMBINED]] : i32
! CHECK: }

subroutine acc_reduction_add_int(a, b)
  integer :: a(100)
  integer :: i, b

  !$acc loop reduction(+:b)
  do i = 1, 100
    b = b + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_int(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_i32 -> %[[B]] : !fir.ref<i32>) {

subroutine acc_reduction_add_float(a, b)
  real :: a(100), b
  integer :: i

  !$acc loop reduction(+:b)
  do i = 1, 100
    b = b + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_float(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_f32 -> %[[B]] : !fir.ref<f32>)
