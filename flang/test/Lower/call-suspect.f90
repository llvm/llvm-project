! Note: flang will issue warnings for the following subroutines. These
! are accepted regardless to maintain backwards compatibility with
! other Fortran implementations.

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPs1() {
! CHECK: fir.convert %{{.*}} : ((!fir.boxchar<1>) -> ()) -> ((!fir.ref<f32>) -> ())

! Pass a REAL by reference to a subroutine expecting a CHARACTER
subroutine s1
  call s3(r)
end subroutine s1

! CHECK-LABEL: func @_QPs2(
! CHECK: fir.convert %{{.*}} : ((!fir.boxchar<1>) -> ()) -> ((!fir.ref<f32>) -> ())

! Pass a REAL, POINTER data reference to a subroutine expecting a CHARACTER
subroutine s2(p)
  real, pointer :: p
  call s3(p)
end subroutine s2

! CHECK-LABEL: func @_QPs3(
! CHECK-SAME: !fir.boxchar<1>
subroutine s3(c)
  character(8) c
end subroutine s3
