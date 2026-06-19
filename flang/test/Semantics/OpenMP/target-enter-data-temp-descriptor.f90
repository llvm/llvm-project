! RUN: %flang -fsyntax-only -fopenmp -fopenmp-version=52 -Werror -Wno-experimental-option %s 2>&1 | FileCheck --allow-empty %s --check-prefix=NOWARN
! RUN: %flang -fsyntax-only -fopenmp -fopenmp-version=52 -Wopenmp-target-enter-data-local-descriptor -Wno-experimental-option %s 2>&1 | FileCheck %s --check-prefix=EMIT52
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52 -Wopenmp-target-enter-data-local-descriptor -Werror -Wno-experimental-option

! NOWARN-NOT: warning:
! NOWARN-NOT: Semantic errors in
! EMIT52: warning: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
! EMIT52-NOT: Semantic errors in

! Check for warning when mapping variables with temporary stack descriptors
! (assumed-shape, assumed-rank, local allocatables, local pointers) on
! TARGET ENTER DATA without a corresponding TARGET EXIT DATA in the same scope.

subroutine test_assumed_shape_warning(arr)
  integer, intent(inout) :: arr(:)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_shape_2d_warning(arr)
  integer, intent(inout) :: arr(:,:)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_rank_warning(arr)
  integer, intent(inout) :: arr(..)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_local_pointer_warning()
  integer, pointer :: local_ptr(:)
  allocate(local_ptr(100))
  !WARNING: The map of 'local_ptr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: local_ptr)
  deallocate(local_ptr)
end subroutine

subroutine test_local_allocatable_warning()
  integer, allocatable :: local_arr(:)
  allocate(local_arr(100))
  !WARNING: The map of 'local_arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: local_arr)
  deallocate(local_arr)
end subroutine

module test_module
contains
  subroutine test_module_procedure_warning(arr)
    integer, intent(inout) :: arr(:)
    !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
    !$omp target enter data map(to: arr)
  end subroutine

  subroutine test_module_procedure_with_exit(arr)
    integer, intent(inout) :: arr(:)
    !$omp target enter data map(to: arr)
    !$omp target exit data map(from: arr)
  end subroutine
end module

subroutine test_internal_scope_warning(outer)
  integer, intent(inout) :: outer(:)
  !WARNING: The map of 'outer' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-target-enter-data-local-descriptor]
  !$omp target enter data map(to: outer)
contains
  subroutine inner(inner_arr)
    integer, intent(inout) :: inner_arr(:)
    !$omp target enter data map(to: inner_arr)
    !$omp target exit data map(from: inner_arr)
  end subroutine
end subroutine

! Test cases where warnings should not be emitted, the test_errors.py script
! should fail if we emit errors for these that are not checked, so no need to
! verify with an explicit check.

subroutine test_pointer_dummy_no_warning(ptr)
  integer, pointer, intent(inout) :: ptr(:)
  !$omp target enter data map(to: ptr)
end subroutine

subroutine test_allocatable_dummy_no_warning(arr)
  integer, allocatable, intent(inout) :: arr(:)
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_with_exit_data(arr)
  integer, intent(inout) :: arr(:)
  !$omp target enter data map(to: arr)
  !$omp target exit data map(from: arr)
end subroutine

subroutine test_explicit_shape_no_warning(arr, n)
  integer, intent(in) :: n
  integer, intent(inout) :: arr(n)
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_size_no_warning(arr)
  integer, intent(inout) :: arr(*)
  !$omp target enter data map(to: arr(1:10))
end subroutine

subroutine test_local_allocatable_with_exit()
  integer, allocatable :: local_arr(:)
  allocate(local_arr(100))
  !$omp target enter data map(to: local_arr)
  !$omp target exit data map(from: local_arr)
  deallocate(local_arr)
end subroutine

subroutine test_saved_local_allocatable_no_warning()
  integer, allocatable, save :: local_arr(:)
  !$omp target enter data map(to: local_arr)
end subroutine

module test_saved_module
  integer, allocatable :: saved_arr(:)
contains
  subroutine test_module_allocatable_no_warning()
    !$omp target enter data map(to: saved_arr)
  end subroutine
end module
