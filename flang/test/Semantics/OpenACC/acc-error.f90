! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check parser specific error for OpenACC


subroutine test(a, n)
    integer :: a(n)
    !ERROR: expected OpenACC directive
    !$acc p
    integer :: i,j
 
    i = 0
    !ERROR: expected OpenACC directive
    !$acc p
  end subroutine

subroutine test2(a, n)
  integer :: a(n)
  integer :: i

  !$acc parallel
  !$acc loop
  DO i = 1, n
  END DO
  !$acc end parallel
  !WARN: Misplaced OpenACC end directive
  !$acc end loop

end subroutine
