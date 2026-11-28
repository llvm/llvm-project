! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK-E
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-OMP
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OMP

! Test in mixed way, i.e., combination of Fortran free source form
! and free source form with conditional compilation sentinel.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} mixed_form1
! CHECK-E:{{^}}      i = 1 &
! CHECK-E:{{^}}!$    +100&
! CHECK-E:{{^}}!$   &+ 1000&
! CHECK-E:{{^}}     &+ 10 + 1&
! CHECK-E:{{^}}!$   & +100000&
! CHECK-E:{{^}}     &0000 + 1000000
! CHECK-OMP: i=1001001112_4
! CHECK-NO-OMP: i=1010011_4
subroutine mixed_form1()
   i = 1 &
  !$ +100&
  !$&+ 1000&
   &+ 10 + 1&
  !$& +100000&
   &0000 + 1000000
end subroutine

! Testing continuation lines in only Fortran Free form Source
! CHECK-LABEL: {{SUBROUTINE|subroutine}} mixed_form2
! CHECK-E:{{^}}      i = 1 +10 +100 + 1000 + 10000
! CHECK-OMP: i=11111_4
! CHECK-NO-OMP: i=11111_4
subroutine mixed_form2()
   i = 1 &
   +10 &
   &+100
   & + 1000 &
   + 10000
end subroutine

! Testing continuation line in only free source form conditional compilation sentinel.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} mixed_form3
! CHECK-E:{{^}}!$    i=0
! CHECK-E:{{^}}!$    i = 1 &
! CHECK-E:{{^}}!$   & +10 &
! CHECK-E:{{^}}!$   &+100&
! CHECK-E:{{^}}!$    +1000
! CHECK-OMP: i=0_4
! CHECK-OMP: i=1111_4
! CHECK-NO-OMP-NOT: i=0_4
subroutine mixed_form3()
   !$ i=0
   !$ i = 1 &
   !$ & +10 &
   !$&+100&
   !$ +1000
end subroutine

! CHECK-LABEL: {{SUBROUTINE|subroutine}} regression
! CHECK-E:{{^}}!$    real x, &
! CHECK-E:{{^}}      stop
! CHECK-OMP: REAL x, stop
! CHECK-NO-OMP-NOT: REAL x,
subroutine regression
!$ real x, &
 stop
end

! Free source form conditional compilation sentinel on a continuation line
! that is immediately followed by the continued text, with no white space or
! ampersand in between (https://github.com/llvm/llvm-project/issues/198771).
! CHECK-LABEL: {{SUBROUTINE|subroutine}} mixed_form4
! CHECK-OMP: i=1111111_4
! CHECK-NO-OMP: i=1010001_4
subroutine mixed_form4()
!$ i = 0
 i = 1&
   !$ +10 &
!$+100&
!$&+ 1000&
 &+ 10000&
!$& +100000&
 & +1000000
end subroutine

! Same, but continuing a conditional compilation line.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} mixed_form5
! CHECK-OMP: i=21_4
! CHECK-NO-OMP-NOT: i=21_4
subroutine mixed_form5()
!$ i = 1&
!$+10&
!$*2
end subroutine

! An initial line still needs white space after the sentinel, so these
! are comments.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} initial_line
! CHECK-OMP-NOT: i=5_4
! CHECK-OMP-NOT: j=6_4
subroutine initial_line()
!$i = 5
!$+j = 6
end subroutine

! Comment lines between continued conditional compilation lines.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} cond_comment
! CHECK-OMP: i=5_4
subroutine cond_comment()
!$ i = 1 &
! comment
!$!comment
!$ + 4
end subroutine

! Same as above, but with blanks between the sentinel and the comment.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} cond_comment_blanks
! CHECK-OMP: i=7_4
subroutine cond_comment_blanks()
!$ i = 1 &
!$ !comment
!$	!comment
!$ + 6
end subroutine

! Digits following the sentinel on a continuation line.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} digits
! CHECK-OMP: i=1234_4
! CHECK-NO-OMP: i=12_4
subroutine digits()
  i = 12&
!$34&
  &+0
end subroutine

! A name longer than any sentinel following the sentinel on a continuation
! line.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} longname
! CHECK-OMP: i=1_4+abcdefghij
! CHECK-NO-OMP: i=1_4{{$}}
subroutine longname()
  integer :: abcdefghij
  i = 1 &
!$+abcdefghij &
  + 0
end subroutine

! A digit following the sentinel on an initial line: this is a comment.
! CHECK-LABEL: {{SUBROUTINE|subroutine}} initial_digit
! CHECK-OMP-NOT: k=7_4
subroutine initial_digit()
!$1 k = 7
end subroutine
