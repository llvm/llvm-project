! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=61

! OpenMP 5.2 3.2.1 excludes substrings from variable, extended, and locator
! list items, independently of the restrictions specific to each clause.
module substring_lists
  character(8) :: device_string
  !ERROR: A substring cannot appear in a ENTER clause
  !$omp declare target enter(device_string(2:3))
  character(8) :: thread_string
  !$omp threadprivate(thread_string)
contains
  subroutine data_sharing(c, a)
    character(8) :: c, a(2)
    type box
      character(8) :: field
    end type
    type(box) :: b
    integer :: i

    !ERROR: A substring cannot appear in a SHARED clause
    !$omp parallel shared(c(2:3))
    !$omp end parallel
    !ERROR: A substring cannot appear in a PRIVATE clause
    !$omp parallel private(a(1)(2:3))
    !$omp end parallel
    !ERROR: A substring cannot appear in a FIRSTPRIVATE clause
    !$omp parallel firstprivate(b%field(2:3))
    !$omp end parallel
    !ERROR: A substring cannot appear in a LASTPRIVATE clause
    !$omp parallel do lastprivate(c(2:3))
    do i = 1, 2
    end do
    !$omp end parallel do
    !ERROR: A substring cannot appear in a COPYIN clause
    !$omp parallel copyin(thread_string(2:3))
    !$omp end parallel
    !$omp parallel private(c)
    !$omp single
    !ERROR: A substring cannot appear in a COPYPRIVATE clause
    !$omp end single copyprivate(c(2:3))
    !$omp end parallel
  end subroutine

  subroutine locators(c, a)
    character(8) :: c, a(2)
    !ERROR: A substring cannot appear in a MAP clause
    !$omp target map(tofrom: c(2:3))
    !$omp end target
    !ERROR: A substring cannot appear in a TO clause
    !$omp target update to(a(1)(2:3))
    !ERROR: A substring cannot appear in a FROM clause
    !$omp target update from(c(2:3))
    ! DEPEND retains its existing substring extension.
    !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
    !$omp task depend(in: c(2:3))
    !$omp end task
    !$omp task depend(in: a(1)(2:3))
    !$omp end task
    ! AFFINITY retains its warning cases and explicit-substring errors.
    !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
    !$omp task affinity(c(2:3))
    !$omp end task
    !ERROR: Substrings are not allowed on AFFINITY clause
    !$omp task affinity(a(1)(2:3))
    !$omp end task

    ! Whole characters and character array sections remain valid list items.
    !$omp parallel shared(c, a)
    !$omp end parallel
    !$omp target map(tofrom: c, a(1:2))
    !$omp end target
    !$omp task depend(in: c, a(1:2)) affinity(c, a(1:2))
    !$omp end task
  end subroutine
end module
