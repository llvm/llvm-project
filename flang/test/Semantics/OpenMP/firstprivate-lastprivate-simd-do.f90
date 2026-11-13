! RUN: %flang_fc1 -fopenmp -Wportability %s 2>&1 | FileCheck %s

! Diagnose overlaps once per variable, independent of clause order.
subroutine overlaps
  integer :: i, a, b, c
  common /block/ a, b
  !$omp parallel
  !$omp do simd firstprivate(a) &
  !CHECK: Variable 'a' appears in both FIRSTPRIVATE and LASTPRIVATE clauses on a DO SIMD construct; 'a' may be uninitialised in the associated structured block [-Wportability]
  !$omp& lastprivate(a)
  do i = 1, 10
    a = i
    b = i
  end do
  !$omp end do simd

  !CHECK: Variable 'a' appears in both FIRSTPRIVATE and LASTPRIVATE clauses on a DO SIMD construct; 'a' may be uninitialised in the associated structured block [-Wportability]
  !CHECK: Variable 'b' appears in both FIRSTPRIVATE and LASTPRIVATE clauses on a DO SIMD construct; 'b' may be uninitialised in the associated structured block [-Wportability]
  !$omp do simd firstprivate(/block/) lastprivate(a, b)
  do i = 1, 10
    a = i
    b = i
  end do
  !$omp end do simd

  !CHECK-NOT: Variable {{'.*'}} appears in both FIRSTPRIVATE and LASTPRIVATE clauses on a DO SIMD construct; {{'.*'}} may be uninitialised in the associated structured block [-Wportability]
  !$omp do simd firstprivate(a)
  do i = 1, 10
    a = i
    b = i
  end do
  !$omp end do simd

  !CHECK-NOT: Variable {{'.*'}} appears in both FIRSTPRIVATE and LASTPRIVATE clauses on a DO SIMD construct; {{'.*'}} may be uninitialised in the associated structured block [-Wportability]
  !$omp do simd lastprivate(a)
  do i = 1, 10
    a = i
    b = i
  end do
  !$omp end do simd  
  !$omp end parallel

end subroutine
