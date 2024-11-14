! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! XFAIL: *

! OpenMP Version 4.5
! 2.8.1 simd Construct
! An ordered construct with the simd clause is the only OpenMP construct
! that can be encountered during execution of a simd region.

program omp_simd
  integer i, j, k
  integer, allocatable :: a(:)

  allocate(a(10))

  !$omp simd
  do i = 1, 10
    !ERROR: Invalid OpenMP construct inside simd region
    !$omp single
    a(i) = i
    !$omp end single
  end do
  !$omp end simd

  print *, a

end program omp_simd
