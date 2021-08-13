! Verify that the compiler driver correctly processes `-fsyntax-only`. By
! default it will try to run code-generation, but that's not supported yet. We
! don't need to test the frontend driver here - it runs `-fsyntax-only` by
! default.

!-----------
! RUN LINES
!-----------
! RUN: not %flang -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=FSYNTAX_ONLY
! RUN: not %flang  %s 2>&1 | FileCheck %s --check-prefix=NO_FSYNTAX_ONLY

!-----------------
! EXPECTED OUTPUT
!-----------------
! FSYNTAX_ONLY: IF statement is not allowed in IF statement
! FSYNTAX_ONLY_NEXT: Semantic errors in {{.*}}syntax-only.f90

! NO_FSYNTAX_ONLY: error: code-generation is not available yet

!-------
! INPUT
!-------
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
