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

subroutine affinity_section_component(n)
  integer, intent(in) :: n
  type t
    integer :: x
  end type
  type(t) :: a(n)
  !ERROR: Structure components are not allowed in an AFFINITY clause
  !$omp task affinity(a(1:n)%x)
    a(1)%x = 1
  !$omp end task
end subroutine

subroutine affinity_iterator_component(n)
  integer, intent(in) :: n
  integer :: i
  type t
    integer :: x
  end type
  type(t) :: a(n)
  !ERROR: Structure components are not allowed in an AFFINITY clause
  !$omp task affinity(iterator(i = 1:n) : a(i)%x)
    a(1)%x = 1
  !$omp end task
end subroutine

subroutine affinity_iterator_section_component(n)
  integer, intent(in) :: n
  integer :: i
  type t
    integer :: x
  end type
  type(t) :: a(n)
  !ERROR: Structure components are not allowed in an AFFINITY clause
  !$omp task affinity(iterator(i = 1:n) : a(i:i+1)%x)
    a(1)%x = 1
  !$omp end task
end subroutine
