! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test declare reduction without initializer clause for derived types
! with default component initialization.

! CHECK-LABEL: omp.declare_reduction @add_reduction_byref_rec__QMtypesTshape
! CHECK-SAME:    : !fir.ref<!fir.type<_QMtypesTshape{center:!fir.type<_QMtypesTpoint{x:f32,y:f32}>,radius:f32}>>
! CHECK:       init {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<!fir.type<_QMtypesTshape{{.*}}>>, %[[PRIV:.*]]: !fir.ref<!fir.type<_QMtypesTshape{{.*}}>>):
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.type<_QMtypesTshape{{.*}}>
! CHECK:         fir.zero_bits !fir.type<_QMtypesTpoint{x:f32,y:f32}>
! CHECK:         fir.insert_value %[[UNDEF]], {{.*}}["center"
! CHECK:         arith.constant 0.000000e+00 : f32
! CHECK:         fir.insert_value {{.*}}["radius"
! CHECK:         fir.store {{.*}} to %[[PRIV]]
! CHECK:         omp.yield(%[[PRIV]]
! CHECK:       } combiner {

! CHECK-LABEL: omp.declare_reduction @add_reduction_byref_rec__QMtypesTpoint
! CHECK-SAME:    : !fir.ref<!fir.type<_QMtypesTpoint{x:f32,y:f32}>>
! CHECK:       init {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<!fir.type<_QMtypesTpoint{{.*}}>>, %[[PRIV2:.*]]: !fir.ref<!fir.type<_QMtypesTpoint{{.*}}>>):
! CHECK:         fir.zero_bits !fir.type<_QMtypesTpoint{x:f32,y:f32}>
! CHECK:         fir.store
! CHECK:         omp.yield(%[[PRIV2]]
! CHECK:       } combiner {

! CHECK-LABEL: func.func @_QQmain
! CHECK:       omp.wsloop {{.*}} reduction(byref @add_reduction_byref_rec__QMtypesTpoint
! CHECK:       omp.wsloop {{.*}} reduction(byref @add_reduction_byref_rec__QMtypesTshape

module types
  implicit none

  type :: point
    real :: x , y
  end type point

  ! Nested derived type — tests recursive isSimpleReductionType
  type :: shape
    type(point) :: center
    real :: radius = 0.0
  end type shape

  interface operator(+)
    module procedure add_points, add_shapes
  end interface

contains

  pure function add_points(a, b) result(res)
    type(point), intent(in) :: a, b
    type(point) :: res
    res%x = a%x + b%x
    res%y = a%y + b%y
  end function

  pure function add_shapes(a, b) result(res)
    type(shape), intent(in) :: a, b
    type(shape) :: res
    res%center = add_points(a%center, b%center)
    res%radius = a%radius + b%radius
  end function

end module types

program test_no_init_derived
  use types
  implicit none

  type(point) :: p
  type(shape) :: s
  integer :: i

  !$omp declare reduction(+: point: omp_out = omp_out + omp_in)
  !$omp declare reduction(+: shape: omp_out = omp_out + omp_in)

  p = point(0.0, 0.0)
  s = shape(point(0.0, 0.0), 0.0)

  ! Test point reduction without initializer
  !$omp parallel do reduction(+: p)
  do i = 1, 10
    p = p + point(1.0, 2.0)
  end do
  !$omp end parallel do

  ! Test nested shape reduction without initializer
  !$omp parallel do reduction(+: s)
  do i = 1, 10
    s = s + shape(point(1.0, 2.0), 3.0)
  end do
  !$omp end parallel do

  print *, p%x, p%y
  print *, s%center%x, s%center%y, s%radius
end program
