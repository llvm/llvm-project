! RUN: %flang_fc1 -fsyntax-only -fno-automatic %s 2>&1 | FileCheck %s --allow-empty
! Checks that -fno-automatic implies the SAVE attribute.
! This same subroutine appears in test save01.f90 where it is an
! error case due to the absence of both SAVE and -fno-automatic.
subroutine foo
  integer, target :: t
  !CHECK-NOT: error:
  integer, pointer :: p => t
end
