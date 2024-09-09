! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine test_scan()
 integer x, y, k, z
 
 !ERROR: Orphaned `omp scan` directives are prohibited; perhaps you forgot to enclose the directive in to a worksharing loop, a worksharing loop simd or a simd directive.
 !$omp scan inclusive(x)
 !$omp parallel do simd
 do k = 1, n
 !ERROR: UNTIED clause is not allowed on the SCAN directive
   !$omp scan untied
 end do
 
 !$omp parallel do simd
 do k = 1, n
 !ERROR: Exactly one of `exclusive` or `inclusive` clause is expected
   !$omp scan
 end do

!$omp parallel do simd reduction(inscan,+: x, y)
 do k = 1, n
 !ERROR: Exactly one of `exclusive` or `inclusive` clause is expected
   !$omp scan inclusive(x) exclusive(y)
 end do

!ERROR: List item y must appear in 'inclusive' or 'exclusive' clause of an enclosed scan directive
!$omp parallel do simd reduction(inscan,+: x, y)
 do k = 1, n
 !ERROR: Exactly one of `exclusive` or `inclusive` clause is expected
 !ERROR: List item z must appear in 'reduction' clause with the 'inscan' modifier of the parent directive
   !$omp scan inclusive(x) exclusive(z)
 end do
end subroutine
