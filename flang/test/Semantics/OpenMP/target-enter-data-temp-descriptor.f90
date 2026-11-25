! RUN: %python %S/../test_errors.py %s %flang -fopenmp -Werror -fopenmp-version=52 -Wno-experimental-option
! RUN: %flang -fsyntax-only -fopenmp -fopenmp-version=52 -Wno-openmp-usage -Wno-experimental-option %s 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
! QUIET-NOT: {{.}}
! RUN: %flang -fsyntax-only -fopenmp -fopenmp-version=52 -Wopenmp-usage -Wno-experimental-option %s 2>&1 | FileCheck %s --check-prefix=GUIDANCE --implicit-check-not=ref_ptee
! GUIDANCE: warning: Mapping allocatable component 'this%arr' may leave its descriptor mapped when only the allocated storage is released
! GUIDANCE: A TARGET EXIT DATA through a nonallocatable assumed-shape dummy does not identify the original descriptor; release the component directly or through an ALLOCATABLE dummy

! Check for warning when mapping variables with temporary stack descriptors
! (assumed-shape, assumed-rank, local allocatables, local pointers) on
! TARGET ENTER DATA without a corresponding TARGET EXIT DATA in the same scope.

subroutine test_assumed_shape_warning(arr)
  integer, intent(inout) :: arr(:)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_shape_2d_warning(arr)
  integer, intent(inout) :: arr(:,:)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_assumed_rank_warning(arr)
  integer, intent(inout) :: arr(..)
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
end subroutine

subroutine test_local_pointer_warning()
  integer, pointer :: local_ptr(:)
  allocate(local_ptr(100))
  !WARNING: The map of 'local_ptr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: local_ptr)
  deallocate(local_ptr)
end subroutine

subroutine test_local_allocatable_warning()
  integer, allocatable :: local_arr(:)
  allocate(local_arr(100))
  !WARNING: The map of 'local_arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: local_arr)
  deallocate(local_arr)
end subroutine

module test_module
contains
  subroutine test_module_procedure_warning(arr)
    integer, intent(inout) :: arr(:)
    !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
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

subroutine test_iterator_assumed_shape_no_warning(arr, n)
  integer, intent(inout) :: arr(:)
  integer, intent(in) :: n
  integer :: i
  !$omp target enter data map(iterator(i = 1:n), to: arr(i))
end subroutine

subroutine test_iterator_local_allocatable_no_warning()
  integer, allocatable :: local_arr(:)
  integer :: i
  allocate(local_arr(100))
  !$omp target enter data map(iterator(i = 1:100), to: local_arr(i))
  deallocate(local_arr)
end subroutine

subroutine test_iterator_mixed_object_warning(arr, other, n)
  integer, intent(inout) :: arr(:)
  integer, intent(inout) :: other(n)
  integer, intent(in) :: n
  integer :: i
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(iterator(i = 1:n), to: arr, other(i))
end subroutine

subroutine test_plain_enter_iterator_exit_warning(arr, n)
  integer, intent(inout) :: arr(:)
  integer, intent(in) :: n
  integer :: i
  !WARNING: The map of 'arr' may include a descriptor that is created locally. Mapping this descriptor without an appropriate TARGET EXIT DATA in the same scope may result in the device retaining an invalid descriptor reference [-Wopenmp-usage]
  !$omp target enter data map(to: arr)
  !$omp target exit data map(iterator(i = 1:n), from: arr(i))
end subroutine

! A component descriptor belongs to the containing object. In particular,
! INTENT(OUT) does not imply a temporary copy of a scalar derived-type dummy.
! An ordinary assumed-shape exit dummy cannot identify that descriptor.
module component_descriptor_maps
  type :: field_t
    real, allocatable :: arr(:)
  end type
contains
  subroutine init(this)
    type(field_t), intent(out) :: this
    allocate(this%arr(4))
    !WARNING: Mapping allocatable component 'this%arr' may leave its descriptor mapped when only the allocated storage is released [-Wopenmp-usage]
    !$omp target enter data map(alloc: this%arr)
  end subroutine

  subroutine finish(this)
    type(field_t), intent(inout) :: this
    call device_free(this%arr)
    deallocate(this%arr)
  end subroutine

  subroutine device_free(data)
    real, contiguous, intent(inout) :: data(:)
    !$omp target exit data map(delete: data)
  end subroutine

  subroutine matched_component_maps(this)
    type(field_t), intent(inout) :: this
    !$omp target enter data map(to: this%arr)
    !$omp target exit data map(from: this%arr)
  end subroutine
end module
