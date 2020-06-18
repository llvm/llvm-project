! RUN: %S/test_errors.sh %s %t %f18
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
    assign 3 to lab
    write(*,fmt=lab) ! ok
    write(*,fmt=implicitlab3) ! ok
1   continue
3   format('yes')
  end subroutine test
end program
