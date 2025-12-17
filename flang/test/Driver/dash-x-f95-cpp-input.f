program main
  print *, __FILE__, __LINE__
end

! This test verifies that `flang`'s `-x` options behave like `gfortran`'s.
! Specifically:
! - `-x f95` should process the file based on its extension unless overridden.
! - `-x f95-cpp-input` should behave like `-x f95` but with preprocessing
!   (`-cpp`) enabled unless overridden.

! ---
! Ensure the file is treated as fixed-form unless explicitly set otherwise
! ---
! RUN: not %flang -Werror -fsyntax-only -x f95 -cpp %s 2>&1 | FileCheck --check-prefix=SCAN-ERROR %s
! RUN: not %flang -Werror -fsyntax-only -x f95-cpp-input %s 2>&1 | FileCheck --check-prefix=SCAN-ERROR %s

! SCAN-ERROR: error: Could not scan

! RUN: %flang -Werror -fsyntax-only -x f95 -cpp -ffree-form %s 2>&1 | FileCheck --check-prefix=NO-SCAN-ERROR --allow-empty %s
! RUN: %flang -Werror -fsyntax-only -x f95-cpp-input -ffree-form %s 2>&1 | FileCheck --check-prefix=NO-SCAN-ERROR --allow-empty %s

! NO-SCAN-ERROR-NOT: error

! ---
! Ensure `-cpp` is not enabled by default unless explicitly requested
! ---
! RUN: not %flang -Werror -fsyntax-only -x f95 -ffree-form %s 2>&1 | FileCheck --check-prefix=SEMA-ERROR %s
! RUN: not %flang -Werror -fsyntax-only -x f95-cpp-input -nocpp -ffree-form %s 2>&1 | FileCheck --check-prefix=SEMA-ERROR %s

! SEMA-ERROR: error: Semantic errors

! RUN: %flang -Werror -fsyntax-only -x f95 -cpp -ffree-form %s 2>&1 | FileCheck --check-prefix=NO-SEMA-ERROR --allow-empty %s
! RUN: %flang -Werror -fsyntax-only -x f95-cpp-input -ffree-form %s 2>&1 | FileCheck --check-prefix=NO-SEMA-ERROR --allow-empty %s

! NO-SEMA-ERROR-NOT: error
