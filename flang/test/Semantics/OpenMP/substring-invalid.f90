! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

! Recognizing malformed substrings must not discard diagnostics and mark
! symbols erroneous: later references to the same symbol used to crash.
subroutine scalar_substrings(s)
  character(8) :: s
  !ERROR: A substring cannot have a stride
  !$omp parallel private(s(::2))
  !$omp end parallel
  !ERROR: A substring cannot have a stride
  !$omp parallel private(s(1::2))
  !$omp end parallel
  !ERROR: A substring cannot have a stride
  !$omp parallel private(s(:8:2))
  !$omp end parallel
  !ERROR: A substring cannot have a stride
  !$omp parallel private(s(1:8:2))
  !$omp end parallel

  !$omp parallel shared(s)
  !$omp end parallel
end subroutine

subroutine component_substrings()
  type box
    character(8) :: field
    character(8) :: array(8)
  end type
  type(box) :: b(2)

  ! Malformed component substrings must also retain their diagnostics.
  !ERROR: A substring cannot have a stride
  !$omp parallel private(b(1)%field(1:4:2))
  !$omp end parallel
  !ERROR: A substring cannot have a stride
  !$omp parallel private(b(1:2)%field(::2))
  !$omp end parallel

  ! A character array section is not a substring.
  !$omp target map(tofrom: b(1)%array(1:4))
  !$omp end target
end subroutine

! Other malformed references still exercise symbol-error recovery.
subroutine scalar_array_section(i)
  integer :: i
  !ERROR: 'i' is not an array
  !$omp parallel private(i(:))
  !$omp end parallel
  !$omp parallel shared(i)
  !$omp end parallel
end subroutine

! Invalid bounds must not also produce a clause-specific substring error.
subroutine invalid_substring_bounds(s)
  character(8) :: s
  !ERROR: Substring must begin at 1 or later, not 0
  !$omp parallel private(s(0:4))
  !$omp end parallel
  !ERROR: Substring must end at 8 or earlier, not 9
  !$omp parallel private(s(1:9))
  !$omp end parallel
end subroutine
