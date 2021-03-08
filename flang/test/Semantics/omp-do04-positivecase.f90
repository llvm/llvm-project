! RUN: %S/test_symbols.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop Constructs

!DEF: /omp_do1 MainProgram
program omp_do1
  !DEF: /omp_do1/i ObjectEntity INTEGER(4)
  !DEF: /omp_do1/j ObjectEntity INTEGER(4)
  !DEF: /omp_do1/k (OmpThreadprivate) ObjectEntity INTEGER(4)
  !DEF: /omp_do1/n (OmpThreadprivate) ObjectEntity INTEGER(4)
  integer i, j, k, n
  !$omp threadprivate (k,n)
  !$omp do
  !DEF: /omp_do1/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_do1/j
    do j=1,10
      print *, "Hello"
    end do
  end do
  !$omp end do
end program omp_do1
