! RUN: bbc %s -o - | FileCheck %s

! Test support of non standard features regarding common blocks:
! - A named common that appears with different storage sizes
! - A blank common that is initialized
! - A common block that is initialized outside of a BLOCK DATA.

! CHECK-LABEL: fir.global @_QB : tuple<i32, !fir.array<8xi8>> {
! CHECK:  %[[undef:.*]] = fir.undefined tuple<i32, !fir.array<8xi8>>
! CHECK:  %[[init:.*]] = fir.insert_value %[[undef]], %c42{{.*}}, [0 : index] : (tuple<i32, !fir.array<8xi8>>, i32) -> tuple<i32, !fir.array<8xi8>>
! CHECK:  fir.has_value %[[init]] : tuple<i32, !fir.array<8xi8>>

! CHECK-LABEL: fir.global @_QBa : tuple<i32, !fir.array<8xi8>> {
! CHECK:  %[[undef:.*]] = fir.undefined tuple<i32, !fir.array<8xi8>>
! CHECK:  %[[init:.*]] = fir.insert_value %[[undef]], %c42{{.*}}, [0 : index] : (tuple<i32, !fir.array<8xi8>>, i32) -> tuple<i32, !fir.array<8xi8>>
! CHECK:  fir.has_value %[[init]] : tuple<i32, !fir.array<8xi8>>


subroutine first_appearance
  real :: x, y, xa, ya
  common // x, y
  common /a/ xa, ya
  call foo(x, xa)
end subroutine

subroutine second_appearance
  real :: x, y, z, xa, ya, za
  common // x, y, z
  common /a/ xa, ya, za
  call foo(x, xa)
end subroutine

subroutine third_appearance
  integer :: x = 42, xa = 42
  common // x
  common /a/ xa
end subroutine
