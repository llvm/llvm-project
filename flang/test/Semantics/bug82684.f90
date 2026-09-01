! Out-of-bounds constant subscripts are errors by default, but
! -fout-of-bounds-subscripts reduces them to warnings, since such a
! reference is nonconforming only if it is actually executed.
! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
! RUN: %flang_fc1 -fsyntax-only -fout-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
! RUN: %flang_fc1 -fsyntax-only -fout-of-bounds-subscripts -Wno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=CHECK-SILENT --allow-empty %s

module m
  integer, parameter :: n_dims = 2
  type :: grid_type
    real :: cells(1000, n_dims)
  end type
 contains
  ! This subprogram is never called, so it never renders the program
  ! nonconforming, but that can't be proven here.
  subroutine init_3Dgrid(node)
    type(grid_type), intent(inout) :: node
    !CHECK-ERROR: error: subscript 3 is greater than upper bound 2 for dimension 2 of array
    !CHECK-WARNING: warning: subscript 3 is greater than upper bound 2 for dimension 2 of array [-Wout-of-bounds-subscripts]
    !CHECK-SILENT-NOT: subscript 3
    node%cells(:,3) = 3.0
  end subroutine
  subroutine lower(node)
    type(grid_type), intent(inout) :: node
    !CHECK-ERROR: error: subscript 0 is less than lower bound 1 for dimension 2 of array
    !CHECK-WARNING: warning: subscript 0 is less than lower bound 1 for dimension 2 of array [-Wout-of-bounds-subscripts]
    !CHECK-SILENT-NOT: subscript 0
    node%cells(:,0) = 0.0
  end subroutine
end module
