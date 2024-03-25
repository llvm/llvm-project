! RUN: %flang_fc1 -fopenmp -E %s | FileCheck %s
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
      end
