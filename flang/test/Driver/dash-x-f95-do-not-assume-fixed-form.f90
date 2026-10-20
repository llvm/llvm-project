! This test verifies that using `-x f95` does not cause the driver to assume
! this file is in fixed-form.

program main
  print *, "Hello, World!"
end

! RUN: %flang -### -x f95 %s 2>&1 | FileCheck --check-prefix=PRINT-PHASES %s
! PRINT-PHASES-NOT: -ffixed-form

! RUN: %flang -Werror -fsyntax-only -x f95 %s 2>&1 | FileCheck --check-prefix=COMPILE --allow-empty %s
! COMPILE-NOT: error
