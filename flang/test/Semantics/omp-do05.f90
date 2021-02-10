! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct restrictions on single directive.


program omp_do

  integer n
  integer i,j
  !$omp do
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end do

  !$omp parallel default(shared)
  !$omp do
  do i = 1, n
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    call work(i, 1)
    !$omp end single
  end do
  !$omp end do
  !$omp end parallel

end program omp_do
