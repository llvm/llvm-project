! RUN: %python %S/../test_errors.py %s %flang -fopenacc

program openacc_combined_loop
  implicit none
  integer :: i

  i = 1

  !ERROR: A DO loop must follow the PARALLEL LOOP directive
  !$acc parallel loop
  i = 1

  !ERROR: A DO loop must follow the KERNELS LOOP directive
  !$acc kernels loop
  i = 1

  !ERROR: A DO loop must follow the SERIAL LOOP directive
  !$acc serial loop
  i = 1

  !$acc parallel loop
  do 10 i=0, n
  10 continue

  !$acc kernels loop
  do 20 i=0, n
  20 continue

  !$acc serial loop
  do 30 i=0, n
  30 continue

end
