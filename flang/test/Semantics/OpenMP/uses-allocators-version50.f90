! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50

! In OpenMP 5.0 the USES_ALLOCATORS clause only accepts the
! "allocator[(traits-array)]" list syntax, and [5.0:175] requires a
! non-predefined allocator to specify traits.

module uses_allocators_50_traits
  use omp_lib
  type(omp_alloctrait), parameter :: module_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
end module

subroutine uses_allocators_50
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! The list syntax is the only syntax in 5.0, so it is not diagnosed.
  !$omp target uses_allocators(my_alloc(tr), other_alloc(tr))
  x = 1
  !$omp end target

  !$omp target uses_allocators(omp_default_mem_alloc)
  x = 2
  !$omp end target

  !ERROR: A non-predefined allocator 'my_alloc' in a USES_ALLOCATORS clause must have traits specified in OpenMP v5.0
  !$omp target uses_allocators(my_alloc)
  x = 3
  !$omp end target

  !WARNING: 'traits-array' modifier is not supported in OpenMP v5.0, try -fopenmp-version=52
  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 4
  !$omp end target

  ! Accepting the newer syntax with a warning does not skip the remaining
  ! checks on the specification.
  !WARNING: 'traits-array' modifier is not supported in OpenMP v5.0, try -fopenmp-version=52
  !ERROR: The traits array must be a named constant array
  !$omp target uses_allocators(traits(1): my_alloc)
  x = 5
  !$omp end target

  !WARNING: 'mem-space' modifier is not supported in OpenMP v5.0, try -fopenmp-version=52
  !ERROR: A non-predefined allocator 'my_alloc' in a USES_ALLOCATORS clause must have traits specified in OpenMP v5.0
  !$omp target uses_allocators(memspace(omp_default_mem_space): my_alloc)
  x = 6
  !$omp end target

  !ERROR: A non-predefined allocator 'omp_null_allocator' in a USES_ALLOCATORS clause must be a variable
  !$omp target uses_allocators(omp_null_allocator(tr))
  x = 7
  !$omp end target
end subroutine

subroutine uses_allocators_50_association
  use omp_lib
  use uses_allocators_50_traits
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

subroutine uses_allocators_50_rename
  use omp_lib, only: renamed_predef => omp_const_mem_alloc
  integer :: x

  ! Predefined recognition follows the entity before 6.0, so a rename of the
  ! intrinsic omp_lib allocator is still predefined and needs no traits. The
  ! same rename is rejected at 6.0; see uses-allocators-version60.f90.
  !$omp target uses_allocators(renamed_predef)
  x = 1
  !$omp end target
end subroutine
