!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine entry_between()
  integer :: a(10), i
  !$omp metadirective &
!ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  entry alternate_entry()
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine format_between()
  integer :: a(10), i
  !$omp metadirective &
!ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
100 format(i0)
  do i = 1, 10
    a(i) = i
  end do
end subroutine
