! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61 -Werror -Wno-experimental-option

! Check for OpenMP 6.1+ specific warning that includes ref_ptee suggestion
! when mapping variables with temporary stack descriptors on TARGET ENTER DATA
! without a corresponding TARGET EXIT DATA.

subroutine test_assumed_shape_warning(arr)
  integer, intent(inout), dimension(:,:) :: arr(:)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference. To avoid mapping the descriptor utilize OpenMP's ref_ptee reference modifier to map just the data [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_rank_warning(arr)
  integer, intent(inout) :: arr(..)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference. To avoid mapping the descriptor utilize OpenMP's ref_ptee reference modifier to map just the data [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_local_allocatable_warning()
  integer, allocatable :: local_arr(:)
  allocate(local_arr(100))
  !WARNING: The map of 'local_arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference. To avoid mapping the descriptor utilize OpenMP's ref_ptee reference modifier to map just the data [-Wopenmp-usage]
  !$omp target enter data map(to: local_arr)
  deallocate(local_arr)
end subroutine

subroutine test_local_pointer_warning()
  integer, pointer :: local_ptr(:)
  allocate(local_ptr(100))
  !WARNING: The map of 'local_ptr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference. To avoid mapping the descriptor utilize OpenMP's ref_ptee reference modifier to map just the data [-Wopenmp-usage]
  !$omp target enter data map(to: local_ptr)
  deallocate(local_ptr)
end subroutine

module test_module
contains
  subroutine test_module_procedure_warning(arr)
    integer, intent(inout) :: arr(:)
    !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference. To avoid mapping the descriptor utilize OpenMP's ref_ptee reference modifier to map just the data [-Wopenmp-usage]
    !$omp target enter data map(to: arr)
  end subroutine

  subroutine test_module_procedure_with_exit(arr)
    integer, intent(inout) :: arr(:)
    !$omp target enter data map(to: arr)
    !$omp target exit data map(from: arr)
  end subroutine
end module

! Test cases where warnings should not be emitted, the test_errors.py script
! should fail if we emit errors for these that are not checked, so no need to
! verify with an explicit check.

subroutine test_ref_ptee_no_warning(arr)
  integer, intent(inout) :: arr(:)
  !$omp target enter data map(ref_ptee, to: arr)
end subroutine

subroutine test_ref_ptr_no_warning(arr)
  integer, intent(inout) :: arr(:)
  !$omp target enter data map(ref_ptr, to: arr)
end subroutine

subroutine test_ref_ptr_ptee_no_warning(arr)
  integer, intent(inout) :: arr(:)
  !$omp target enter data map(ref_ptr_ptee, to: arr)
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

subroutine test_local_allocatable_with_exit()
  integer, allocatable :: local_arr(:)
  allocate(local_arr(100))
  !$omp target enter data map(to: local_arr)
  !$omp target exit data map(from: local_arr)
  deallocate(local_arr)
end subroutine

subroutine test_allocatable_dummy_no_warning(arr)
  integer, allocatable, intent(inout) :: arr(:)
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_pointer_dummy_no_warning(ptr)
  integer, pointer, intent(inout) :: ptr(:)
  !$omp target enter data map(to: ptr)
end subroutine
