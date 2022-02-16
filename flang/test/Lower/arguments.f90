! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub1(a, b)
  integer, intent(in) :: a
  logical :: b
end

! Check that arguments are correctly set and no local allocation is happening.
! CHECK-LABEL: func @_QPsub1(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "a"}, %{{.*}}: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"})
! CHECK-NOT:     fir.alloc
! CHECK:         return

subroutine sub2(i)
  integer :: i(2, 5)
end

! CHECK-LABEL: func @_QPsub2(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<2x5xi32>>{{.*}})

subroutine sub3(i)
  real :: i(2)
end

! CHECK-LABEL: func @_QPsub3(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<2xf32>>{{.*}})

integer function fct1(a, b)
  integer, intent(in) :: a
  logical :: b
end

! CHECK-LABEL: func @_QPfct1(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "a"}, %{{.*}}: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) -> i32

real function fct2(i)
  integer :: i(2, 5)
end

! CHECK-LABEL: func @_QPfct2(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<2x5xi32>> {fir.bindc_name = "i"}) -> f32

function fct3(i)
  real :: i(2)
end

! CHECK-LABEL: func @_QPfct3(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "i"}) -> f32
