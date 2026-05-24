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
