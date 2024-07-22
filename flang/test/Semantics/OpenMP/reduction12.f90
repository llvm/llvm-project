! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! OpenMP 5.2: Section 5.5.5 : A procedure pointer must not appear in a
! reduction clause.

  procedure(foo), pointer :: ptr
  integer :: i
  ptr => foo
!ERROR: A procedure pointer 'ptr' must not appear in a REDUCTION clause.
!$omp do reduction (+ : ptr)
  do i = 1, 10
  end do
contains
  subroutine foo
  end subroutine
end
