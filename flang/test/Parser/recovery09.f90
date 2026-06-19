! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Verify that a data component definition appearing after CONTAINS in a
! derived type gives a clear error instead of misleading
! "expected 'FINAL'/'GENERIC'/'PROCEDURE'" messages.

module m
  implicit none

  type, public :: t1
     real :: x
   contains
     procedure, public :: init
! CHECK: error: component definition must precede CONTAINS in a derived type
! CHECK-NEXT: {{.*}}integer, public :: n(3) = 1
     integer, public :: n(3) = 1
! CHECK: error: component definition must precede CONTAINS in a derived type
! CHECK-NEXT: {{.*}}real, pointer, dimension(:,:,:), public :: gpoint => null()
     real, pointer, dimension(:,:,:), public :: gpoint => null()
! CHECK-NOT: expected 'FINAL'
! CHECK-NOT: expected 'GENERIC'
! CHECK-NOT: expected 'PROCEDURE'
  end type t1

contains
  subroutine init(this)
    class(t1), intent(inout) :: this
  end subroutine init
end module m
