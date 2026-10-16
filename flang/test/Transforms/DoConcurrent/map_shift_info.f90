! Tests that descriptors declared with a `fir.shift` (explicit lower bounds)
! are re-declared with a rebuilt `fir.shift` inside the target region.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine assumed_shape_lbound(a, b)
  implicit none
  real, intent(in) :: a(5:, :)
  real, intent(out) :: b(:, :)
  integer :: j
  do concurrent (j = 1:size(b, 2))
    b(:, j) = a(:, j)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPassumed_shape_lbound
! CHECK: %[[LB0_MAP:.*]] = omp.map.info {{.*}} name("_QFassumed_shape_lboundEa.start_idx.dim0")
! CHECK: %[[LB1_MAP:.*]] = omp.map.info {{.*}} name("_QFassumed_shape_lboundEa.start_idx.dim1")
! CHECK: omp.target {{.*}}%[[LB0_MAP]] -> %[[LB0_ARG:[^,]+]], %[[LB1_MAP]] -> %[[LB1_ARG:[^,]+]]
! CHECK-DAG: %[[LB0:.*]] = fir.load %[[LB0_ARG]] : !fir.ref<index>
! CHECK-DAG: %[[LB1:.*]] = fir.load %[[LB1_ARG]] : !fir.ref<index>
! CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
! CHECK: %[[SHIFT:.*]] = fir.shift %[[LB0]], %[[LB1]] : (index, index) -> !fir.shift<2>
! CHECK: hlfir.declare %[[BOX]](%[[SHIFT]]) uniq_name("_QFassumed_shape_lboundEa")

subroutine associate_pointer_component(p, b)
  implicit none
  type t
    real, pointer :: values(:)
  end type
  type(t), intent(in) :: p
  real, intent(out) :: b(:)
  integer :: i
  associate(v => p%values)
    do concurrent (i = 1:size(b))
      b(i) = v(i)
    end do
  end associate
end subroutine

! CHECK-LABEL: func.func @_QPassociate_pointer_component
! CHECK: omp.target
! CHECK: %[[SHIFT:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
! CHECK: hlfir.declare %{{.*}}(%[[SHIFT]]) uniq_name("_QFassociate_pointer_componentEv")