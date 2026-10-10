! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! Scalar character components remain substrings even with array-valued parents.
! Substring strides are invalid whether or not either bound is present.

subroutine affinity_substring_strides(a, s)
  type t
    character(8) :: field
    character(8) :: array(8)
  end type
  type(t) :: a(8)
  character(8) :: s

  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(s(::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(s(1::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(s(:8:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(a(1)%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(a(1:2)%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(a(1:2)%field(::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(a%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task affinity(iterator(i=1:2): a(1:i)%field(1:4:2))
  !$omp end task

  ! Without a step, these substrings remain accepted as an extension.
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task affinity(a(1)%field(1:4))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task affinity(a(1:2)%field(1:4))
  !$omp end task

  ! An array character component uses array section rules instead.
  !ERROR: 'array' in AFFINITY clause must not specify a stride
  !$omp task affinity(a(1)%array(1:4:2))
  !$omp end task
  !$omp task affinity(a(1)%array(1:4))
  !$omp end task
end subroutine

subroutine depend_substring_strides(a, s)
  type t
    character(8) :: field
    character(8) :: array(8)
  end type
  type(t) :: a(8)
  character(8) :: s

  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: s(::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: s(1::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: s(:8:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: a(1)%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: a(1:2)%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: a(1:2)%field(::2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(in: a%field(1:4:2))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(iterator(i=1:2), in: a(1:i)%field(1:4:2))
  !$omp end task

  ! Without a step, these substrings remain accepted as an extension.
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(in: a(1)%field(1:4))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(in: a(1:2)%field(1:4))
  !$omp end task

  ! An array character component uses array section rules instead.
  !ERROR: 'array' in DEPEND clause must not specify a stride
  !$omp task depend(in: a(1)%array(1:4:2))
  !$omp end task
  !$omp task depend(in: a(1)%array(1:4))
  !$omp end task
end subroutine
