! Check driver handling of -f[no-]out-of-bounds-subscripts.  An out-of-range
! constant subscript is a warning by default; -fno-out-of-bounds-subscripts
! makes it an error.  The last of the two spellings on the command line wins.

! RUN: %flang -fsyntax-only %s 2>&1 | FileCheck --check-prefix=WARN %s
! RUN: %flang -fsyntax-only -fout-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=WARN %s
! RUN: not %flang -fsyntax-only -fno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=ERROR %s
! RUN: not %flang -fsyntax-only -fout-of-bounds-subscripts -fno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=ERROR %s
! RUN: %flang -fsyntax-only -fno-out-of-bounds-subscripts -fout-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=WARN %s
! RUN: %flang -fsyntax-only -Wno-out-of-bounds-subscripts %s 2>&1 | FileCheck --check-prefix=SILENT --allow-empty %s

module m
  real :: a(5)
 contains
  ! This subprogram is never called.
  subroutine never_called()
    !WARN: warning: subscript 6 is greater than upper bound 5 for dimension 1 of array [-Wout-of-bounds-subscripts]
    !ERROR: error: subscript 6 is greater than upper bound 5 for dimension 1 of array
    !SILENT-NOT: subscript 6
    a(6) = 0.
  end subroutine
end module
