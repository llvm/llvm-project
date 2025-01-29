! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Check for ORDER([order-modifier :]concurrent) clause on OpenMP constructs

subroutine test_order()
 integer :: i, j = 1
 !$omp do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end do
end subroutine

!CHECK: - file:         {{.*}}
!CHECK:   line:         9
!CHECK:   construct:    do
!CHECK:   clauses:
!CHECK:     - clause:   order
!CHECK:       details:  concurrent

subroutine test_order_reproducible()
 integer :: i, j = 1
 !$omp simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end simd
end subroutine

!CHECK: - file:         {{.*}}
!CHECK:   line:         25
!CHECK:   construct:    simd
!CHECK:   clauses:
!CHECK:     - clause:   order
!CHECK:       details:  'reproducible:concurrent'

subroutine test_order_unconstrained()
 integer :: i, j = 1
 !$omp target teams distribute parallel do simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target teams distribute parallel do simd
end subroutine

!CHECK: - file:         {{.*}}
!CHECK:   line:         41
!CHECK:   construct:    target teams distribute parallel do simd
!CHECK:   clauses:
!CHECK:     - clause:   order
!CHECK:       details:  'unconstrained:concurrent'
