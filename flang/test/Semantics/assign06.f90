! RUN: %python %S/test_errors.py %s %flang_fc1
! Test ASSIGN statement, assigned GOTO, and assigned format labels
! (see subclause 8.2.4 in Fortran 90 (*not* 2018!)

program main
  call test(0)
 contains
  subroutine test(n)
    integer, intent(in) :: n
    integer :: lab
    integer(kind=1) :: badlab1
    real :: badlab2
    integer :: badlab3(1)
    real, pointer :: badlab4(:) ! not contiguous
    real, pointer, contiguous :: oklab4(:)
    assign 1 to lab ! ok
    assign 1 to implicitlab1 ! ok
    !ERROR: 'badlab1' must be a default integer scalar variable
    assign 1 to badlab1
    !ERROR: 'badlab2' must be a default integer scalar variable
    assign 1 to badlab2
    !ERROR: 'badlab3' must be a default integer scalar variable
    assign 1 to badlab3
    !ERROR: 'test' must be a default integer scalar variable
    assign 1 to test
    if (n==1) goto lab ! ok
    if (n==1) goto implicitlab2 ! ok
    !ERROR: 'badlab1' must be a default integer scalar variable
    if (n==1) goto badlab1
    !ERROR: 'badlab2' must be a default integer scalar variable
    if (n==1) goto badlab2
    !ERROR: 'badlab3' must be a default integer scalar variable
    if (n==1) goto badlab3
    if (n==1) goto lab(1) ! ok
    if (n==1) goto lab,(1) ! ok
    if (n==1) goto lab(1,1) ! ok
    assign 3 to lab ! ok
    write(*,fmt=lab) ! ok
    write(*,fmt=implicitlab3) ! ok
    !ERROR: Format expression must be default character or default scalar integer
    write(*,fmt=badlab1)
    !ERROR: Format expression must be default character or default scalar integer
    write(*,fmt=z'feedface')
    !Legacy extension cases
    write(*,fmt=badlab2)
    write(*,fmt=badlab3)
    !ERROR: Format expression must be a simply contiguous array if not scalar
    write(*,fmt=badlab4)
    write(*,fmt=badlab5) ! ok legacy extension
1   continue
3   format('yes')
  end subroutine test
end program
