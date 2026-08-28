! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

! OpenMP 5.1 keeps the pre-5.2 USES_ALLOCATORS syntax and the [5.1:203]
! requirement that a non-predefined allocator specifies traits.

module uses_allocators_51_traits
  use omp_lib
  type(omp_alloctrait), parameter :: module_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
end module

subroutine uses_allocators_51
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !$omp target uses_allocators(my_alloc(tr), other_alloc(tr))
  x = 1
  !$omp end target

  !ERROR: A non-predefined allocator 'my_alloc' in a USES_ALLOCATORS clause must have traits specified in OpenMP v5.1
  !$omp target uses_allocators(my_alloc)
  x = 2
  !$omp end target

  !WARNING: 'traits-array' modifier is not supported in OpenMP v5.1 on USES_ALLOCATORS clause, try -fopenmp-version=52
  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 3
  !$omp end target

  ! Accepting the newer syntax with a warning does not skip the remaining
  ! checks on the specification.
  !WARNING: 'traits-array' modifier is not supported in OpenMP v5.1 on USES_ALLOCATORS clause, try -fopenmp-version=52
  !ERROR: The traits array must be a named constant array
  !$omp target uses_allocators(traits(1): my_alloc)
  x = 4
  !$omp end target

  !ERROR: A predefined allocator 'omp_default_mem_alloc' in a USES_ALLOCATORS clause cannot have modifiers or traits specified
  !$omp target uses_allocators(omp_default_mem_alloc(tr))
  x = 5
  !$omp end target
end subroutine

subroutine uses_allocators_51_association
  use omp_lib
  use uses_allocators_51_traits
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: host_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !ERROR: The traits array 'module_tr' must be defined in the same scope as the construct
  !$omp target uses_allocators(my_alloc(module_tr))
  x = 1
  !$omp end target

  call inner
contains
  subroutine inner
    !ERROR: The traits array 'host_tr' must be defined in the same scope as the construct
    !$omp target uses_allocators(my_alloc(host_tr))
    x = 2
    !$omp end target
  end subroutine
end subroutine

subroutine uses_allocators_51_rename
  use omp_lib, only: renamed_predef => omp_const_mem_alloc
  integer :: x

  ! Predefined recognition follows the entity before 6.0, so a rename of the
  ! intrinsic omp_lib allocator is still predefined and needs no traits. The
  ! same rename is rejected at 6.0; see uses-allocators-version60.f90.
  !$omp target uses_allocators(renamed_predef)
  x = 1
  !$omp end target
end subroutine
