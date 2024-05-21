! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5

program p
integer :: y
!ERROR: Variable 'y' may not appear on both MAP and FIRSTPRIVATE clauses on a TARGET construct
!$omp target map(y) firstprivate(y)
y = y + 1
!$omp end target
!ERROR: Variable 'y' may not appear on both MAP and FIRSTPRIVATE clauses on a TARGET SIMD construct
!$omp target simd map(y) firstprivate(y)
do i=1,1
  y = y + 1
end do
!$omp end target simd

end program p
