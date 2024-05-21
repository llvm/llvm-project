! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine omp_order()
 integer :: i, j = 1
 !ERROR: At most one ORDER clause can appear on the SIMD directive
 !$omp simd order(concurrent) order(concurrent)
 do i=1,10
  j = j + 1
 end do
end subroutine omp_order
