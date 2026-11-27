! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50

subroutine omp_order()
 integer :: i, j = 1
 !ERROR: At most one ORDER clause can appear on SIMD directive
 !$omp simd order(concurrent) order(concurrent)
 do i=1,10
  j = j + 1
 end do
end subroutine omp_order
