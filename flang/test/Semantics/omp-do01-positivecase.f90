! RUN: %S/test_symbols.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a firstprivate directive.
! A positive case

!DEF: /omp_do MainProgram
program omp_do
  !DEF: /omp_do/i ObjectEntity INTEGER(4)
  integer i

  !$omp do  firstprivate(k)
  !DEF: /omp_do/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    print *, "Hello"
  end do
  !$omp end do

end program omp_do
