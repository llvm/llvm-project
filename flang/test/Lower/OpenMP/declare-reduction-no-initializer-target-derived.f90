! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test declare reduction without initializer clause for derived type with
! default component values, used in a target offload region.
! The init region must initialize components using the type's default values.

! CHECK-LABEL: omp.declare_reduction @add_pts
! CHECK-SAME:    : !fir.ref<!fir.type<_QFTpoint{x:f32,y:f32}>>
! CHECK:       init {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<!fir.type<_QFTpoint{{.*}}>>,
! CHECK-SAME:       %[[PRIV:.*]]: !fir.ref<!fir.type<_QFTpoint{{.*}}>>):
! CHECK-NOT:     fir.call @_FortranAInitialize
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.type<_QFTpoint{{.*}}>
! CHECK:         fir.zero_bits f32
! CHECK:         fir.insert_value %[[UNDEF]], {{.*}}["x"
! CHECK:         arith.constant 0.000000e+00 : f32
! CHECK:         fir.insert_value {{.*}}["y"
! CHECK:         fir.store {{.*}} to %[[PRIV]]
! CHECK:         omp.yield(%[[PRIV]]
! CHECK:       } combiner {

program main
    implicit none

    type :: Point
        real :: x
        real :: y = 0.0
    end type Point

    integer :: i
    type(Point) :: total

    !$omp declare reduction(add_pts : Point : &
    !$omp&   merge_points(omp_out, omp_in))

    total = Point(0.0, 0.0)

    !$omp target teams distribute parallel do reduction(add_pts: total) map(tofrom: total)
    do i = 1, 100
        total%x = total%x + 1.0
        total%y = total%y + 2.0
    end do
    !$omp end target teams distribute parallel do

    print *, "Final Point X:", total%x
    print *, "Final Point Y:", total%y

contains
    subroutine merge_points(out_p, in_p)
        !$omp declare target
        type(Point), intent(inout) :: out_p
        type(Point), intent(in)    :: in_p
        out_p%x = out_p%x + in_p%x
        out_p%y = out_p%y + in_p%y
    end subroutine
end program main
