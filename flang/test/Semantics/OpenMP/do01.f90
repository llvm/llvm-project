! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a firstprivate directive.

program omp_do
  integer i, j, k

  !ERROR: Loop iteration variable with a predetermined data sharing attribute cannot appear in a FIRSTPRIVATE clause
  !$omp do firstprivate(k,i)
  !BECAUSE: 'i' is an iteration variable of an affected loop
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end program omp_do
