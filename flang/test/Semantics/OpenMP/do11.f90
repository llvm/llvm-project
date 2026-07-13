! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop Constructs

!DEF: /OMP_DO MainProgram
program OMP_DO
  !DEF: /OMP_DO/i ObjectEntity INTEGER(4)
  !DEF: /OMP_DO/j ObjectEntity INTEGER(4)
  !DEF: /OMP_DO/k ObjectEntity INTEGER(4)
  integer i, j, k
  !$omp do
  !DEF: /OMP_DO/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /OMP_DO/j
    do j=1,10
      !REF: /OMP_DO/OtherConstruct1/i
      !REF: /OMP_DO/j
      print *, "it", i, j
    end do
  end do
  !$omp end do
end program OMP_DO

!DEF: /omp_do2 (Subroutine)Subprogram
subroutine omp_do2
  !DEF: /omp_do2/i ObjectEntity INTEGER(4)
  !DEF: /omp_do2/k ObjectEntity INTEGER(4)
  integer :: i = 0, k
  !$omp do
  !DEF: /omp_do2/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_do2/OtherConstruct1/i
    print *, "it", i
  end do
  !$omp end do
end subroutine omp_do2
