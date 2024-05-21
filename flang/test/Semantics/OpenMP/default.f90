!RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.3.1 default Clause
program omp_default
  integer :: a(10), b(10), c(10),i,k
  !ERROR: At most one DEFAULT clause can appear on the PARALLEL directive
  !$omp parallel default(shared), default(private)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel

  !ERROR: At most one DEFAULT clause can appear on the TASK directive
  !$omp task default(shared), default(none), shared(a,b,c,k,i)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end task

  !ERROR: At most one DEFAULT clause can appear on the TASKLOOP directive
  !$omp taskloop default(shared), default(private)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end taskloop

  !ERROR: At most one DEFAULT clause can appear on the TEAMS directive
  !$omp teams default(shared), default(none), shared(i,a,b,k,c)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end teams

end program omp_default
