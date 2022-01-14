! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Check warning on multiple SAVE attribute specifications
subroutine saves
  save x
  save y
  !CHECK: SAVE attribute was already specified on 'y'
  integer, save :: y
  integer, save :: z
  !CHECK: SAVE attribute was already specified on 'x'
  !CHECK: SAVE attribute was already specified on 'z'
  save x,z
end

