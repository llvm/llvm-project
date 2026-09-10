! An out-of-range constant subscript is accepted with a warning, because such a
! reference is nonconforming only if it is actually executed.
! -fno-out-of-bounds-subscripts restores the error.
! A cosubscript is not covered: it stays an error in every mode, so every RUN
! line below expects a failing compilation.
! RUN: not %flang_fc1 -fsyntax-only -fcoarray %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
! RUN: not %flang_fc1 -fsyntax-only -fcoarray -fno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
! RUN: not %flang_fc1 -fsyntax-only -fcoarray -Wno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=CHECK-SILENT %s

module m
  integer, parameter :: n_dims = 2
  type :: grid_type
    real :: cells(1000, n_dims)
  end type
  real :: a(5)
  integer :: c(3)[2:4,*]
  integer, parameter :: k(2) = [1, 2]
 contains
  ! This subprogram is never called, so it never renders the program
  ! nonconforming, but that can't be proven here.
  subroutine init_3Dgrid(node)
    type(grid_type), intent(inout) :: node
    !CHECK-WARNING: warning: subscript 3 is greater than upper bound 2 for dimension 2 of array [-Wout-of-bounds-subscripts]
    !CHECK-ERROR: error: subscript 3 is greater than upper bound 2 for dimension 2 of array
    !CHECK-SILENT-NOT: subscript 3
    node%cells(:,3) = 3.0
  end subroutine
  subroutine lower(node)
    type(grid_type), intent(inout) :: node
    !CHECK-WARNING: warning: subscript 0 is less than lower bound 1 for dimension 2 of array [-Wout-of-bounds-subscripts]
    !CHECK-ERROR: error: subscript 0 is less than lower bound 1 for dimension 2 of array
    !CHECK-SILENT-NOT: subscript 0
    node%cells(:,0) = 0.0
  end subroutine
  ! An array section endpoint is validated by the same code path.
  subroutine section()
    !CHECK-WARNING: warning: subscript 6 is greater than upper bound 5 for dimension 1 of array [-Wout-of-bounds-subscripts]
    !CHECK-ERROR: error: subscript 6 is greater than upper bound 5 for dimension 1 of array
    !CHECK-SILENT-NOT: subscript 6
    a(4:6) = 0.
  end subroutine
  ! A cosubscript is NOT covered: its requirement is F'2023 9.6 p2 and it
  ! determines an image index, so it stays an error in every mode.
  subroutine cosubscript()
    !CHECK-WARNING: error: cosubscript 1 is less than lower cobound 2 for codimension 1 of array
    !CHECK-ERROR: error: cosubscript 1 is less than lower cobound 2 for codimension 1 of array
    !CHECK-SILENT: error: cosubscript 1 is less than lower cobound 2 for codimension 1 of array
    c(1)[1,1] = 0
  end subroutine
  ! Nor is a reference to a named constant array, a DATA statement
  ! designator, or a substring; those stay errors in every mode too.
  subroutine still_errors()
    real :: d(10)
    character(4) :: s
    data d(0)/0./
    !CHECK-WARNING: error: Subscript value (0) is out of range on dimension 1 in reference to a constant array value
    !CHECK-ERROR: error: Subscript value (0) is out of range on dimension 1 in reference to a constant array value
    !CHECK-SILENT: error: Subscript value (0) is out of range on dimension 1 in reference to a constant array value
    print *, k(0)
    !CHECK-WARNING: error: Substring must end at 4 or earlier, not 9
    !CHECK-ERROR: error: Substring must end at 4 or earlier, not 9
    !CHECK-SILENT: error: Substring must end at 4 or earlier, not 9
    print *, s(2:9)
  end subroutine
end module
