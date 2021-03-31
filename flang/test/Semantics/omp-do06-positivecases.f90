! RUN: %S/test_symbols.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The ordered clause must be present on the loop construct if any ordered
! region ever binds to a loop region arising from the loop construct.

! A positive case
!DEF: /omp_do MainProgram
program omp_do
  !DEF: /omp_do/i ObjectEntity INTEGER(4)
  !DEF: /omp_do/j ObjectEntity INTEGER(4)
  !DEF: /omp_do/k ObjectEntity INTEGER(4)
  integer i, j, k
  !$omp do  ordered
    !DEF: /omp_do/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do i=1,10
      !$omp ordered
      !DEF: /my_func EXTERNAL (Subroutine) ProcEntity
      call my_func
      !$omp end ordered
    end do
  !$omp end do
end program omp_do
