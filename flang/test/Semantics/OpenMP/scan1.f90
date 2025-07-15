! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine test_scan()
 integer x, y, k, z
 
 !ERROR: Orphaned SCAN directives are prohibited; perhaps you forgot to enclose the directive in to a WORKSHARING LOOP, a WORKSHARING LOOP SIMD or a SIMD directive.
 !ERROR: List item x must appear in REDUCTION clause with the INSCAN modifier of the parent directive
 !$omp scan inclusive(x)
 !$omp parallel do simd
 do k = 1, n
 !ERROR: UNTIED clause is not allowed on the SCAN directive
   !$omp scan untied
 end do
 
 !$omp parallel do simd
 do k = 1, n
 !ERROR: Exactly one of EXCLUSIVE or INCLUSIVE clause is expected
   !$omp scan
 end do

!$omp parallel do simd reduction(inscan,+: x, y)
 do k = 1, n
 !ERROR: Exactly one of EXCLUSIVE or INCLUSIVE clause is expected
   !$omp scan inclusive(x) exclusive(y)
 end do

!ERROR: List item y must appear in EXCLUSIVE or INCLUSIVE clause of an enclosed SCAN directive
!$omp parallel do simd reduction(inscan,+: x, y)
 do k = 1, n
 !ERROR: Exactly one of EXCLUSIVE or INCLUSIVE clause is expected
 !ERROR: List item z must appear in REDUCTION clause with the INSCAN modifier of the parent directive
   !$omp scan inclusive(x) exclusive(z)
 end do
end subroutine
