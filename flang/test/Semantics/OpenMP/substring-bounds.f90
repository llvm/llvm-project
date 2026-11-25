! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=61

! Permitted substring extensions still require valid substring bounds.
! An error in an earlier clause must not suppress later diagnostics.
subroutine substring_bounds(s)
  character(8) :: s
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substring must begin at 1 or later, not 0
  !$omp task depend(in: s(0:4))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substring must end at 8 or earlier, not 9
  !$omp task affinity(s(1:9))
  !$omp end task

  ! Subsequent uses of the same symbol remain valid.
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(in: s(1:4))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task affinity(s(1:4))
  !$omp end task
end subroutine

subroutine component_substring_bounds()
  type box
    character(8) :: field
  end type
  type(box) :: b(2)
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substring must end at 8 or earlier, not 9
  !$omp task depend(in: b(1:2)%field(1:9))
  !$omp end task
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substring must begin at 1 or later, not 0
  !$omp task affinity(b(1:2)%field(0:4))
  !$omp end task
end subroutine

! Explicit substring parse nodes also need validation on DEPEND.
subroutine array_element_substring_bounds(a)
  character(8) :: a(2)
  !ERROR: Substring must begin at 1 or later, not 0
  !$omp task depend(in: a(1)(0:4))
  !$omp end task
  !ERROR: Substring must end at 8 or earlier, not 9
  !$omp task depend(in: a(2)(1:9))
  !$omp end task
  !$omp task depend(in: a(1)(1:4))
  !$omp end task
end subroutine
