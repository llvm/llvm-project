! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test initialization and finalizations of dummy arguments in entry statements.

module m
  type t
  end type
contains
 subroutine test1(x)
   class(t), intent(out) :: x
   entry test1_entry()
 end subroutine
 subroutine test2(x)
   class(t), intent(out) :: x
   entry test2_entry(x)
 end subroutine
end module
! CHECK-LABEL:   func.func @_QMmPtest1_entry(
! CHECK-NOT: Destroy
! CHECK-NOT: Initialize
! CHECK:           return

! CHECK-LABEL:   func.func @_QMmPtest2_entry(
! CHECK: Destroy
! CHECK: Initialize
! CHECK:           return
