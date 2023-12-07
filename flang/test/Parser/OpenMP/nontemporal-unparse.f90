! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck %s

program omp_simd
  integer i
  integer, allocatable :: a(:)

  allocate(a(10))

  !NONTEMPORAL
  !$omp simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd
end program omp_simd
!CHECK-LABEL: PROGRAM omp_simd

!NONTEMPORAL
!CHECK: !$OMP SIMD  NONTEMPORAL(a)
