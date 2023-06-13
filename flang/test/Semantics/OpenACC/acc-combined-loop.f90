! RUN: %python %S/../test_errors.py %s %flang -fopenacc

program openacc_combined_loop
  implicit none
  integer :: i

  i = 1

  !$acc parallel loop
  !ERROR: A DO loop must follow the combined construct
  i = 1

  !$acc kernels loop
  !ERROR: A DO loop must follow the combined construct
  i = 1

  !$acc serial loop
  !ERROR: A DO loop must follow the combined construct
  i = 1

end
