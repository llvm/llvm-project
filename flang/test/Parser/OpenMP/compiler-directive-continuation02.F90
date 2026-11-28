! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s

! Keyword macros that expand to an OpenMP conditional compilation sentinel
! on free source form continuation lines.

#define CC $
#define CCX $x

! The macro expands to the sentinel alone.
! CHECK-LABEL: SUBROUTINE macro_sentinel
! CHECK: i=111_4
subroutine macro_sentinel()
  integer :: i
  i = 1 &
!CC+10&
  &+100
end subroutine

! The expansion is not a sentinel followed by a blank, so the line is a
! comment.
! CHECK-LABEL: SUBROUTINE macro_sentinel_text
! CHECK: i=101_4
subroutine macro_sentinel_text()
  integer :: i
  i = 1 &
!CCX+10&
  &+100
end subroutine
