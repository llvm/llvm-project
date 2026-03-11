! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine affinity_substring()
  character(len=7) :: a(8)
  !ERROR: Substrings are not allowed on AFFINITY clause
  !$omp task affinity(a(2)(2:4))
    a(1) = "abcdefg"
  !$omp end task
end subroutine

subroutine affinity_iterator_substring(n)
  integer, intent(in) :: n
  integer :: i
  character(len=7) :: a(n)
  !ERROR: Substrings are not allowed on AFFINITY clause
  !$omp task affinity(iterator(i = 1:n) : a(i)(2:4))
    a(1) = "abcdefg"
  !$omp end task
end subroutine

subroutine affinity_iterator_section_component(n, m)
  integer, intent(in) :: n, m
  type t
    integer :: x(10)
    integer :: y
  end type
  type(t) :: a(10)

  !ERROR: Subscripts of component 'x' of rank-1 derived type array have rank 1 but must all be scalar
  !$omp task affinity(a(1:n)%x(1:m))
  !$omp end task

  !ERROR: If a list item is an array section, the last part-ref of the list item must have a section subscript list in AFFINITY clause
  !$omp task affinity(a(1:n)%y)
  !$omp end task
end subroutine

subroutine affinity_section_bad_stride(n)
  integer, intent(in) :: n
  integer :: a(n)
  !ERROR: 'a' in AFFINITY clause must have a positive stride
  !$omp task affinity(a(1:n:-1))
  !$omp end task
end subroutine

subroutine affinity_section_zero_size(n)
  integer, intent(in) :: n
  integer :: a(n)
  !ERROR: 'a' in AFFINITY clause is a zero size array section
  !$omp task affinity(a(5:2))
  !$omp end task
end subroutine


subroutine affinity_iterator_noninteger_iv()
  integer :: x(10)
  !ERROR: The iterator variable must be of integer type
  !$omp task affinity(iterator(real :: i = 1:10): x(1))
  !$omp end task
end subroutine

subroutine affinity_iterator_missing_begin()
  integer :: x(10)
  !ERROR: The begin and end expressions in iterator range-specification are mandatory
  !$omp task affinity(iterator(integer :: i = :10:1): x(1))
  !$omp end task
end subroutine

subroutine affinity_iterator_step_zero()
  integer :: x(10)
  !WARNING: The step value in the iterator range is 0
  !$omp task affinity(iterator(integer :: i = 1:10:0): x(1))
  !$omp end task
end subroutine

subroutine affinity_iterator_section_bad_stride(n)
  integer, intent(in) :: n
  integer :: a(n)
  !ERROR: 'a' in AFFINITY clause must have a positive stride
  !$omp task affinity(iterator(i = 1:n): a(i:n:-1))
  !$omp end task
end subroutine

subroutine affinity_substring_like_single_index()
  character(len=7) :: s
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substrings must be in the form parent-string(lb:ub)
  !$omp task affinity(s(2))
  !$omp end task
end subroutine

subroutine affinity_substring_like_step()
  character(len=7) :: s
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(s(2:6:2))
  !$omp end task
end subroutine

subroutine affinity_section_step_zero()
  integer :: a(10)
  !ERROR: 'a' in AFFINITY clause must have a positive stride
  !ERROR: Stride of triplet must not be zero
  !$omp task affinity(a(1:10:0))
  !$omp end task
end subroutine
