! RUN: %python %S/test_errors.py %s %flang_fc1
! Test ASSIGN statement, assigned GOTO, and assigned format labels
! (see subclause 8.2.4 in Fortran 90 (*not* 2018!)

program main
  call test(0)
2 format('no')
 contains
  subroutine test(n)
    !ERROR: Label '4' is not a branch target or FORMAT
4   integer, intent(in) :: n
    integer :: lab
    assign 1 to lab ! ok
    assign 1 to implicitlab1 ! ok
    !ERROR: Label '666' was not found
    assign 666 to lab
    !ERROR: Label '2' was not found
    assign 2 to lab
    assign 4 to lab
    if (n==1) goto lab ! ok
    if (n==1) goto implicitlab2 ! ok
    if (n==1) goto lab(1) ! ok
    if (n==1) goto lab,(1) ! ok
    if (n==1) goto lab(1,1) ! ok
    !ERROR: Label '666' was not found
    if (n==1) goto lab(1,666)
    !ERROR: Label '2' was not found
    if (n==1) goto lab(1,2)
    ! Label 3 is a FORMAT statement in this scope.  It can be assigned and
    ! used as a format, but naming it in the label list of an assigned GOTO
    ! is an error: a FORMAT statement is not a branch target.  The diagnostic
    ! is reported on the FORMAT statement itself, below.
    if (n==1) goto lab(1,3)
    assign 3 to lab
    write(*,fmt=lab) ! ok
    write(*,fmt=implicitlab3) ! ok
1   continue
    !ERROR: Label '3' is not a branch target
3   format('yes')
  end subroutine test
end program
