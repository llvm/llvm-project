! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s

! OpenMP conditional compilation sentinels on continuation lines, immediately
! followed by the continued text. These cases are only valid when OpenMP is
! enabled.

! A name of any length, immediately followed by a comment.
! CHECK-LABEL: SUBROUTINE word_comment1
! CHECK: i=1_4+k
subroutine word_comment1()
  integer :: k, i
  i = 1 + &
!$k!note
end subroutine

! CHECK-LABEL: SUBROUTINE word_comment5
! CHECK: i=1_4+abcde
subroutine word_comment5()
  integer :: abcde, i
  i = 1 + &
!$abcde!note
end subroutine

! CHECK-LABEL: SUBROUTINE word_comment6
! CHECK: i=1_4+abcdef
subroutine word_comment6()
  integer :: abcdef, i
  i = 1 + &
!$abcdef!note
end subroutine

! CHECK-LABEL: SUBROUTINE word_blank_comment
! CHECK: i=1_4+abcde
subroutine word_blank_comment()
  integer :: abcde, i
  i = 1 + &
!$abcde !note
end subroutine

! A continued character literal.
! CHECK-LABEL: SUBROUTINE char_literal
! CHECK: s="abcd"
subroutine char_literal()
  character(4) :: s
  s = 'ab&
!$cd'
end subroutine
