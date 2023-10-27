! RUN: bbc -o - -emit-fir %s | FileCheck %s

! Test lowering of elemental calls with array arguments that use array
! elements as indices.
! As reported in issue #62981, wrong code was being generated in this case.

module test_ops
  implicit none
  interface
    integer elemental function elem_func_i(i)
      integer, intent(in) :: i
    end function
    real elemental function elem_func_r(r)
      real, intent(in) :: r
    end function
  end interface

  integer :: a(3), b(3), v(3), i, j, k, l
  real :: x(2), y(2), u

contains
! CHECK-LABEL: func @_QMtest_opsPcheck_array_elems_as_indices() {
subroutine check_array_elems_as_indices()
! CHECK: %[[A_ADDR:.*]] = fir.address_of(@_QMtest_opsEa) : !fir.ref<!fir.array<3xi32>>
! CHECK: %[[V_ADDR:.*]] = fir.address_of(@_QMtest_opsEv) : !fir.ref<!fir.array<3xi32>>
! CHECK: %[[V:.*]] = fir.array_load %[[V_ADDR]](%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: %[[A:.*]] = fir.array_load %[[A_ADDR]](%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: fir.do_loop
  forall (i=1:3)
! CHECK: %{{.*}} = fir.array_fetch %[[V]], %{{.*}} : (!fir.array<3xi32>, index) -> i32
! CHECK: fir.do_loop
! CHECK: %[[ELEM:.*]] = fir.array_access %[[A]], %{{.*}} : (!fir.array<3xi32>, index) -> !fir.ref<i32>
! CHECK: %{{.*}} = fir.call @_QPelem_func_i(%[[ELEM]]){{.*}} : (!fir.ref<i32>) -> i32
    b(i:i) = elem_func_i(a(v(i):v(i)))
  end forall
end subroutine

! CHECK-LABEL: func @_QMtest_opsPcheck_not_assert() {
subroutine check_not_assert()
  ! Implicit path.
  b = 10 + elem_func_i(a)

  ! Expression as argument, instead of variable.
  forall (i=1:3)
    b(i:i) = elem_func_i(a(i:i) + a(i:i))
  end forall

  ! Nested elemental function calls.
  y = elem_func_r(cos(x))
  y = elem_func_r(cos(x) + u)

  ! Array constructors as elemental function arguments.
  y = atan2( (/ (real(i, 4), i = 1, 2) /), &
             real( (/ (i, i = j, k, l) /), 4) )
end subroutine

end module
