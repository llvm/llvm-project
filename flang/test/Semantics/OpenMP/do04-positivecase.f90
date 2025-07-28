! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop Constructs

!DEF: /OMP_DO1 MainProgram
program OMP_DO1
  !DEF: /OMP_DO1/i ObjectEntity INTEGER(4)
  !DEF: /OMP_DO1/j ObjectEntity INTEGER(4)
  !DEF: /OMP_DO1/k (OmpThreadprivate) ObjectEntity INTEGER(4)
  !DEF: /OMP_DO1/n (OmpThreadprivate) ObjectEntity INTEGER(4)
  integer i, j, k, n
  !$omp threadprivate (k,n)
  !$omp do
  !DEF: /OMP_DO1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /OMP_DO1/j
    do j=1,10
      print *, "Hello"
    end do
  end do
  !$omp end do
end program OMP_DO1
