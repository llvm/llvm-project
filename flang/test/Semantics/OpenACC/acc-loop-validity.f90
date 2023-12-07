! RUN: %python %S/../test_errors.py %s %flang -fopenacc

program openacc_clause_validity

  implicit none

  integer :: i

  i = 0

  !$acc loop
  !ERROR: A DO loop must follow the loop construct
  i = 1

end
