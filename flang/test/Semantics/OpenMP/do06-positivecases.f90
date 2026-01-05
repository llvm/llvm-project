! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The ordered clause must be present on the loop construct if any ordered
! region ever binds to a loop region arising from the loop construct.

! A positive case
!DEF: /OMP_DO MainProgram
program OMP_DO
  !DEF: /OMP_DO/i ObjectEntity INTEGER(4)
  !DEF: /OMP_DO/j ObjectEntity INTEGER(4)
  !DEF: /OMP_DO/k ObjectEntity INTEGER(4)
  integer i, j, k
  !$omp do  ordered
    !DEF: /OMP_DO/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do i=1,10
      !$omp ordered
      !DEF: /my_func EXTERNAL (Subroutine) ProcEntity
      call my_func
      !$omp end ordered
    end do
  !$omp end do
end program OMP_DO
