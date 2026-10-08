! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=60

! OpenMP Version 6.0, Section 8.8: uses_allocators clause.
! 6.0 restates the allocator rule in terms of the *name* of a predefined
! allocator, gives omp_null_allocator and omp_null_mem_space their own
! allowances, and permits more than one clause-argument-specification.
! [6.0:B.2] also removes every feature deprecated in 5.0, 5.1 and 5.2, which
! includes the comma-separated list syntax deprecated by [5.2:181].

module uses_allocators_60_traits
  use omp_lib
  type(omp_alloctrait), parameter :: module_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
end module

subroutine uses_allocators_v60_association
  use omp_lib
  use uses_allocators_60_traits
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: host_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !$omp target uses_allocators(traits(module_tr): my_alloc)
  x = 1
  !$omp end target

  call inner
contains
  subroutine inner
    !$omp target uses_allocators(traits(host_tr): my_alloc)
    x = 2
    !$omp end target
  end subroutine
end subroutine

subroutine uses_allocators_v60_null
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc
  integer :: x

  ! [6.0:315] The clause has no effect for an allocator argument value of
  ! omp_null_allocator, and [6.0:316] exempts it from the variable rule.
  !$omp target uses_allocators(omp_null_allocator)
  x = 1
  !$omp end target

  ! [6.0:315] omp_null_mem_space means omp_default_mem_space here, so it is
  ! accepted even though it is not one of the five predefined memory spaces.
  !$omp target uses_allocators(memspace(omp_null_mem_space): my_alloc)
  x = 2
  !$omp end target
end subroutine

subroutine uses_allocators_v60_name_matching
  use omp_lib, only: omp_allocator_handle_kind
  ! A local named constant that shadows a predefined allocator's spelling.
  integer(omp_allocator_handle_kind), parameter :: omp_const_mem_alloc = 999
  integer :: x

  ! [6.0:315] matches the *name* of a predefined allocator, so this shadowing
  ! declaration is treated as predefined and needs no variable. Before 6.0 the
  ! same program is rejected; see uses-allocators.f90.
  !$omp target uses_allocators(omp_const_mem_alloc)
  x = 1
  !$omp end target
end subroutine

subroutine uses_allocators_v60_rename
  use omp_lib, only: renamed => omp_const_mem_alloc
  integer :: x

  ! [6.0:315] asks whether the identifier in the clause matches the name of a
  ! predefined allocator. 'renamed' does not, so it is a non-predefined
  ! allocator, and being a named constant it violates the variable rule. The
  ! same rename is accepted at 5.0-5.2, where the entity is what matters; see
  ! uses-allocators.f90.
  !ERROR: A non-predefined allocator 'renamed' in a USES_ALLOCATORS clause must be a variable
  !$omp target uses_allocators(renamed)
  x = 1
  !$omp end target
end subroutine

subroutine uses_allocators_v60_legacy_removed
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! [6.0:B.2] The comma-separated list syntax was deprecated in 5.2 and is
  ! removed in 6.0.
  !ERROR: The comma-separated list syntax for the USES_ALLOCATORS clause was deprecated in OpenMP 5.2 and removed in OpenMP 6.0, use 'USES_ALLOCATORS([TRAITS(traits):] allocator)' instead
  !$omp target uses_allocators(my_alloc, other_alloc)
  x = 1
  !$omp end target

  !ERROR: The comma-separated list syntax for the USES_ALLOCATORS clause was deprecated in OpenMP 5.2 and removed in OpenMP 6.0, use 'USES_ALLOCATORS([TRAITS(traits):] allocator)' instead
  !$omp target uses_allocators(my_alloc(tr))
  x = 2
  !$omp end target
end subroutine

subroutine uses_allocators_v60_multiple
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! [6.0:315] permits more than one clause-argument-specification. The
  ! semicolon-separated grammar is not parsed at all (see
  ! flang/test/Parser/OpenMP/uses-allocators-bad-syntax.f90); this shape is
  ! neither grammar, and reaching it reports the implementation gap.
  !ERROR: Multiple allocator specifications in a USES_ALLOCATORS clause are not yet supported
  !$omp target uses_allocators(traits(tr): my_alloc, traits(tr): other_alloc)
  x = 1
  !$omp end target

  ! A single specification is still accepted.
  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 2
  !$omp end target
end subroutine
