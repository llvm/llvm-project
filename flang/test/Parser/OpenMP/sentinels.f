! RUN: %flang_fc1 -E %s | FileCheck %s
! CHECK:      program main
! CHECK:       interface
! CHECK:        subroutine sub(a, b)
! CHECK:!dir$ ignore_tkr a
! CHECK:!dir$ ignore_tkr b
! CHECK:          real(4):: a, b
! CHECK:        end subroutine
! CHECK:       end interface
! CHECK:      PRINT *, "Is '    '"
! CHECK:  123 PRINT *, "Is '123 '"

!@cuf subroutine atcuf;
      program main
       interface
        subroutine sub(a, b)
!dir$ ignore_tkr a
!dir$ ignore_tkr
!dir$+ b
          real(4):: a, b
        end subroutine
       end interface
!
!	comment line
!@fp  PRINT *, "This is a comment line"
!@f p PRINT *, "This is a comment line"
!$    PRINT *, "Is '    '"
!$123 PRINT *, "Is '123 '"
!$ABC PRINT *, "Is 'ABC '"
! $    PRINT *, "This is a comment line 6"
c    $This is a comment line
!0$110This is a comment line

! $ This is a comment line
! $  0This is a comment line
!    &This is a comment line
!  $  This is a comment line
! $   This is a comment line
C $   This is a comment line
c $   his is a comment line
* $   This is a comment line

! Test non-space/non-number char in columns 3-5, for initial lines.
! CHECK-NOT:  "comment"
c$x   PRINT *, "comment"
c$ +  PRINT *, "comment"
c$  * PRINT *, "comment"

! Test non-space/non-number char in columns 3-5, for continuation lines.
! CHECK:      "msg1"
! CHECK-NOT:  "comment"
c$ x  PRINT *, "comment"
c$1  &         , "comment"
c$ x &         , "comment"
c$  +&         , "comment"

c$    PRINT *, "msg1"
c$1  &         , "comment"
c$ x &         , "comment"
c$  +&         , "comment"

! Test valid chars in initial and continuation lines.
! CHECK: !$ 20 PRINT *, "msg2"
! CHECK: !$ & , "msg3"
c$ 20 PRINT *, "msg2"
c$   &         , "msg3"

! CHECK: !$ PRINT *, "msg4",
! CHECK: !$ & "msg5"
c$   0PRINT *, "msg4",
c$   +         "msg5"
      end
