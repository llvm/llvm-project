! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The do-loop cannot be a DO WHILE or a DO loop without loop control.

program omp_do
  integer ::  i = 0,k
  !ERROR: This construct requires a canonical loop nest
  !$omp do
  !BECAUSE: DO WHILE loop is not a valid affected loop
  do while (i <= 10)
    print *, "it",i
    i = i+1
  end do
  !$omp end do

  !ERROR: This construct requires a canonical loop nest
  !$omp do
  !BECAUSE: DO WHILE loop is not a valid affected loop
  do while (i <= 10)
    do while (j <= 10)
      print *, "it",k
      j = j+1
    end do
    i = i+1
  end do
  !$omp end do
end program omp_do
