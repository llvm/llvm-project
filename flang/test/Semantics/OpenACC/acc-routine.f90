! RUN: %python %S/../test_errors.py %s %flang -fopenacc

subroutine sub1(a)
 real, dimension(10) :: a
end subroutine

subroutine sub2(a)
  !$acc routine(sub1) gang(dim:1)
  real, dimension(10) :: a
  call sub1(a)
end subroutine
