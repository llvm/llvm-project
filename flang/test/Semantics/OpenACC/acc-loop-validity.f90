! RUN: %python %S/../test_errors.py %s %flang -fopenacc

program openacc_clause_validity

  implicit none

  integer :: i, n

  i = 0

  !ERROR: A DO loop must follow the LOOP directive
  !$acc loop
  i = 1

  !$acc loop
  do 100 i=0, n
  100 continue

end
