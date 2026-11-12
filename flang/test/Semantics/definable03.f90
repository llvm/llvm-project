! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine sub(j)
  integer, intent(in) :: j
  !ERROR: 'j' may not be used as a DO variable
  !BECAUSE: 'j' is an INTENT(IN) dummy argument
  do j = 1, 10
  end do
end
