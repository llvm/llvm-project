! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s \
! RUN:   --implicit-check-not="expected 'FINAL'" \
! RUN:   --implicit-check-not="expected 'GENERIC'" \
! RUN:   --implicit-check-not="expected 'PROCEDURE'" \
! RUN:   --implicit-check-not="expected 'COMPLEX'" \
! RUN:   --implicit-check-not="expected 'INTEGER'"

! Verify that a statement appearing after CONTAINS in a derived type that is not
! a type-bound procedure binding gives a clear error instead of misleading
! "expected 'FINAL'/'GENERIC'/'PROCEDURE'" or intrinsic-type-spec keyword
! messages.  The --implicit-check-not options above assert, across the whole
! output, that none of those spurious messages reappear.

module m
  implicit none

  ! A data component definition after CONTAINS names the specific problem.
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
  end type t1

  ! A second CONTAINS is not a component definition.
  type :: t2
   contains
   contains
! CHECK: error: expected a type-bound procedure binding (PROCEDURE, GENERIC, or FINAL) after CONTAINS
! CHECK-NEXT: {{.*}}contains
  end type t2

  ! An IMPORT after CONTAINS is likewise not a binding.
  type :: t3
   contains
     import
! CHECK: error: expected a type-bound procedure binding (PROCEDURE, GENERIC, or FINAL) after CONTAINS
! CHECK-NEXT: {{.*}}import
  end type t3

  ! A misplaced subprogram after CONTAINS.
  type :: t4
   contains
     subroutine s
! CHECK: error: expected a type-bound procedure binding (PROCEDURE, GENERIC, or FINAL) after CONTAINS
! CHECK-NEXT: {{.*}}subroutine s
  end type t4

contains
  subroutine init(this)
    class(t1), intent(inout) :: this
  end subroutine init
end module m
