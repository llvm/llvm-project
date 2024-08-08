! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine test_scan()
 integer x, k
 
 !ERROR: Orphaned `omp scan` directives are prohibited; perhaps you forgot to enclose the directive in to a worksharing loop, a worksharing loop simd or a simd directive.
 !$omp scan inclusive(x)
 !$omp parallel do simd reduction(inscan,+: x)
 do k = 1, n
 !ERROR: UNTIED clause is not allowed on the SCAN directive
   !$omp scan untied
 end do
 
 !$omp parallel do simd reduction(inscan,+: x)
 do k = 1, n
 !ERROR: Exactly one of `exclusive` or `inclusive` clause is expected
   !$omp scan
 end do
end subroutine
