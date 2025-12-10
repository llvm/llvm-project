! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --allow-empty
! Ensure no bogus "no explicit type for ..." error on USE-associated
! implicitly-typed COMMON block object in scope with IMPLICIT NONE.
! CHECK-NOT: error:
module m
  common /block/ var
end
subroutine test
  use m
  implicit none
end
