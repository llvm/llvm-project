! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

program main
  implicit none
  integer, parameter :: n = 100
  integer, parameter :: expected = n+2
  integer :: i
  integer :: counter

  counter = 0
  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target map(tofrom:counter)
  counter = counter+1
  !$omp teams distribute reduction(+:counter)
  do i=1, n
     counter = counter+1
  end do
  counter = counter+1
  !$omp end target
 end program
