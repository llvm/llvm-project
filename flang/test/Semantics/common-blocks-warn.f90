! RUN: %flang -fsyntax-only -pedantic 2>&1 %s | FileCheck %s

! Test that a warning is emitted when a named common block appears in
! several scopes with a different storage size.

subroutine size_1
  common x, y
  common /c/ xc, yc
end subroutine

subroutine size_2
  ! OK, blank common size may always differ.
  common x, y, z
  !CHECK: portability: A named COMMON block should have the same size everywhere it appears (12 bytes here)
  common /c/ xc, yc, zc
end subroutine
