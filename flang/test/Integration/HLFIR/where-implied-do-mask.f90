!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -o /dev/null %s

module m
  implicit none
contains
  subroutine test(arr, mask, default)
    real, intent(inout) :: arr(:)
    integer, intent(in) :: mask(:)
    real, intent(in) :: default
    integer :: j
    where([(.not.any(mask == j), j = 1, size(arr))]) arr = default
  end subroutine
end module

! CHECK-LABEL: func.func @_QMmPtest(
! CHECK: hlfir.where
! CHECK:   hlfir.elemental
! CHECK:     hlfir.any
! CHECK:   hlfir.yield
! CHECK: hlfir.region_assign
! CHECK: return
