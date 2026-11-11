! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52

! OpenMP Version 5.2, Section 6.8: uses_allocators clause.

module uses_allocators_traits_module
  use omp_lib
  type(omp_alloctrait), parameter :: module_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
end module

subroutine uses_allocators_ok
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! A predefined allocator without modifiers.
  !$omp target uses_allocators(omp_default_mem_alloc)
  x = 1
  !$omp end target

  ! Since 5.2, a non-predefined allocator may omit its traits.
  !$omp target uses_allocators(my_alloc)
  x = 2
  !$omp end target

  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 3
  !$omp end target

  !$omp target uses_allocators(memspace(omp_high_bw_mem_space): my_alloc)
  x = 4
  !$omp end target

  !$omp target uses_allocators(memspace(omp_const_mem_space), traits(tr): my_alloc)
  x = 5
  !$omp end target

  ! The clause is repeatable on TARGET.
  !$omp target uses_allocators(my_alloc) uses_allocators(omp_const_mem_alloc)
  x = 6
  !$omp end target

  ! The clause reaches the TARGET leaf of a combined construct.
  !$omp target teams uses_allocators(traits(tr): my_alloc)
  x = 7
  !$omp end target teams
end subroutine

subroutine uses_allocators_renamed_type
  use omp_lib, only: renamed_alloctrait => omp_alloctrait, &
      omp_allocator_handle_kind, omp_atk_alignment
  integer(omp_allocator_handle_kind) :: my_alloc
  type(renamed_alloctrait), parameter :: tr(1) = &
      [renamed_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 1
  !$omp end target
end subroutine

subroutine uses_allocators_loop_construct
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: i, x(10)

  ! USES_ALLOCATORS is a data-sharing attribute clause that carries allocator
  ! specifications rather than an object list, so the loop-construct checks
  ! must not treat it as an object-list clause.
  !$omp target teams distribute parallel do uses_allocators(traits(tr): my_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !$omp target simd uses_allocators(my_alloc)
  do i = 1, 10
    x(i) = i
  end do
end subroutine

subroutine uses_allocators_errors
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  integer(omp_memspace_handle_kind) :: my_space
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  type(omp_alloctrait) :: nonconst(1)
  type(omp_alloctrait), parameter :: tr2(1,1) = &
      reshape([omp_alloctrait(omp_atk_alignment, 64)], [1,1])
  real, parameter :: notatrait(1) = [1.0]
  real :: notaninteger
  integer :: x

  ! The allocator must be a base language identifier.
  !ERROR: The allocator in a USES_ALLOCATORS clause must be a base language identifier
  !$omp target uses_allocators(traits(tr): my_alloc + 1)
  x = 1
  !$omp end target

  ! A predefined allocator cannot have modifiers.
  !ERROR: A predefined allocator 'omp_default_mem_alloc' in a USES_ALLOCATORS clause cannot have modifiers or traits specified
  !$omp target uses_allocators(traits(tr): omp_default_mem_alloc)
  x = 2
  !$omp end target

  !ERROR: A predefined allocator 'omp_const_mem_alloc' in a USES_ALLOCATORS clause cannot have modifiers or traits specified
  !$omp target uses_allocators(memspace(omp_const_mem_space): omp_const_mem_alloc)
  x = 3
  !$omp end target

  ! omp_null_allocator is a named constant and is not a predefined allocator.
  !ERROR: A non-predefined allocator 'omp_null_allocator' in a USES_ALLOCATORS clause must be a variable
  !$omp target uses_allocators(omp_null_allocator)
  x = 4
  !$omp end target

  ! The allocator must be a scalar integer.
  !ERROR: Must have INTEGER type, but is REAL(4)
  !$omp target uses_allocators(notaninteger)
  x = 5
  !$omp end target

  ! The memspace-handle must name a predefined memory space.
  !ERROR: The 'mem-space' modifier must name a predefined memory space
  !$omp target uses_allocators(memspace(omp_null_mem_space): my_alloc)
  x = 6
  !$omp end target

  !ERROR: The 'mem-space' modifier must name a predefined memory space
  !$omp target uses_allocators(memspace(my_space): my_alloc)
  x = 7
  !$omp end target

  ! The traits array must be a constant array of OMP_ALLOCTRAIT.
  !ERROR: The traits array 'nonconst' must be a constant array with constant values
  !$omp target uses_allocators(traits(nonconst): my_alloc)
  x = 8
  !$omp end target

  !ERROR: The traits array 'tr2' must be a rank-one array
  !$omp target uses_allocators(traits(tr2): my_alloc)
  x = 9
  !$omp end target

  !ERROR: The traits array 'notatrait' must be of type OMP_ALLOCTRAIT
  !$omp target uses_allocators(traits(notatrait): my_alloc)
  x = 10
  !$omp end target

  ! The allocator must not appear in a data-sharing or data-mapping clause.
  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the PRIVATE clause on the same construct
  !ERROR: Variable 'my_alloc' may not appear on both MAP and PRIVATE clauses on a TARGET construct
  !$omp target uses_allocators(traits(tr): my_alloc) private(my_alloc)
  x = 11
  !$omp end target

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the MAP clause on the same construct
  !$omp target map(tofrom: my_alloc) uses_allocators(traits(tr): my_alloc)
  x = 12
  !$omp end target

  ! A modifier may appear at most once in one allocator specification.
  !ERROR: 'traits-array' modifier cannot occur multiple times
  !$omp target uses_allocators(traits(tr), traits(tr): my_alloc)
  x = 13
  !$omp end target

  !ERROR: 'mem-space' modifier cannot occur multiple times
  !$omp target uses_allocators(memspace(omp_const_mem_space), memspace(omp_high_bw_mem_space): my_alloc)
  x = 131
  !$omp end target

  ! OpenMP 5.2 allows a single allocator specification.
  !ERROR: The USES_ALLOCATORS clause accepts a single allocator specification in OpenMP v5.2
  !$omp target uses_allocators(traits(tr): my_alloc, traits(tr): other_alloc)
  x = 14
  !$omp end target

  ! The clause is not allowed on a non-TARGET construct.
  !ERROR: USES_ALLOCATORS clause is not allowed on PARALLEL directive
  !$omp parallel uses_allocators(my_alloc)
  x = 15
  !$omp end parallel
end subroutine

subroutine uses_allocators_deprecated
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc, other_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! [5.2:181] The "allocator[(traits)]" list syntax is deprecated, but it is
  ! still accepted.
  !PORTABILITY: The comma-separated list syntax for the USES_ALLOCATORS clause has been deprecated in OpenMP 5.2, use 'USES_ALLOCATORS([TRAITS(traits):] allocator)' instead
  !$omp target uses_allocators(my_alloc(tr))
  x = 1
  !$omp end target

  !PORTABILITY: The comma-separated list syntax for the USES_ALLOCATORS clause has been deprecated in OpenMP 5.2, use 'USES_ALLOCATORS([TRAITS(traits):] allocator)' instead
  !$omp target uses_allocators(my_alloc(tr), other_alloc(tr))
  x = 2
  !$omp end target

  ! A bare comma list is also the deprecated syntax.
  !PORTABILITY: The comma-separated list syntax for the USES_ALLOCATORS clause has been deprecated in OpenMP 5.2, use 'USES_ALLOCATORS([TRAITS(traits):] allocator)' instead
  !$omp target uses_allocators(my_alloc, other_alloc)
  x = 3
  !$omp end target

  ! A single bare allocator is the canonical 5.2 syntax and is not deprecated.
  !$omp target uses_allocators(my_alloc)
  x = 4
  !$omp end target
end subroutine

subroutine uses_allocators_traits_association
  use omp_lib
  use uses_allocators_traits_module
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: host_tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! [5.2:182] requires the traits array to be defined in the construct scope.
  !ERROR: The traits array 'module_tr' must be defined in the same scope as the construct
  !$omp target uses_allocators(traits(module_tr): my_alloc)
  x = 1
  !$omp end target

  call inner
contains
  subroutine inner
    !ERROR: The traits array 'host_tr' must be defined in the same scope as the construct
    !$omp target uses_allocators(traits(host_tr): my_alloc)
    x = 2
    !$omp end target
  end subroutine
end subroutine

subroutine uses_allocators_impostor_type
  use omp_lib, only: omp_allocator_handle_kind
  type omp_alloctrait
    integer :: key
    integer :: value
  end type
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: tr(1) = [omp_alloctrait(1, 64)]
  integer :: x

  !ERROR: The traits array 'tr' must be of type OMP_ALLOCTRAIT
  !$omp target uses_allocators(traits(tr): my_alloc)
  x = 1
  !$omp end target
end subroutine

subroutine uses_allocators_combined_conflicts
  use omp_lib
  integer(omp_allocator_handle_kind) :: private_alloc, shared_alloc, fp_alloc
  integer(omp_allocator_handle_kind) :: map_alloc, last_alloc, reduction_alloc
  integer :: i, x(10)

  !$omp target teams distribute parallel do uses_allocators(private_alloc) private(private_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !$omp target teams distribute parallel do uses_allocators(shared_alloc) shared(shared_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the FIRSTPRIVATE clause on the same construct
  !$omp target teams distribute parallel do uses_allocators(fp_alloc) firstprivate(fp_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the MAP clause on the same construct
  !$omp target teams distribute parallel do uses_allocators(map_alloc) map(tofrom: map_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the LASTPRIVATE clause on the same construct
  !$omp target teams distribute parallel do uses_allocators(last_alloc) lastprivate(last_alloc)
  do i = 1, 10
    x(i) = i
  end do

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the REDUCTION clause on the same construct
  !$omp target teams distribute parallel do uses_allocators(reduction_alloc) reduction(+: reduction_alloc)
  do i = 1, 10
    x(i) = i
  end do
end subroutine

subroutine uses_allocators_conflicting_clauses
  use omp_lib
  ! Each case uses its own allocator so that the once-per-symbol
  ! "both MAP and ..." diagnostic of TARGET does not interleave with them.
  integer(omp_allocator_handle_kind) :: fp_alloc, idp_alloc, hda_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the FIRSTPRIVATE clause on the same construct
  !$omp target uses_allocators(traits(tr): fp_alloc) firstprivate(fp_alloc)
  x = 1
  !$omp end target

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the IS_DEVICE_PTR clause on the same construct
  !ERROR: Variable 'idp_alloc' in IS_DEVICE_PTR clause must be of type C_PTR
  !$omp target uses_allocators(traits(tr): idp_alloc) is_device_ptr(idp_alloc)
  x = 2
  !$omp end target

  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the HAS_DEVICE_ADDR clause on the same construct
  !$omp target uses_allocators(traits(tr): hda_alloc) has_device_addr(hda_alloc)
  x = 3
  !$omp end target
end subroutine

subroutine uses_allocators_predefined_identity
  use omp_lib
  use omp_lib, only: renamed_predef => omp_default_mem_alloc
  use omp_lib, only: renamed_space => omp_const_mem_space
  integer(omp_allocator_handle_kind) :: my_alloc
  integer(omp_allocator_handle_kind), parameter :: same_value_as_predef = 1
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

  ! Before 6.0 predefined recognition follows the entity, so a rename of the
  ! intrinsic omp_lib allocator still denotes it and is accepted bare. At 6.0
  ! the clause identifier is what matters and the rename is rejected; see
  ! uses-allocators-version60.f90.
  !$omp target uses_allocators(renamed_predef)
  x = 1
  !$omp end target

  ! ... and, being predefined, it still rejects modifiers.
  !ERROR: A predefined allocator 'renamed_predef' in a USES_ALLOCATORS clause cannot have modifiers or traits specified
  !$omp target uses_allocators(traits(tr): renamed_predef)
  x = 2
  !$omp end target

  ! A named constant that merely shares a predefined allocator's numeric value
  ! is neither the predefined entity nor a predefined name, so the variable
  ! rule applies to it at every version.
  !ERROR: A non-predefined allocator 'same_value_as_predef' in a USES_ALLOCATORS clause must be a variable
  !$omp target uses_allocators(same_value_as_predef)
  x = 3
  !$omp end target

  ! A memory space must use the written name of a predefined memory space.
  !ERROR: The 'mem-space' modifier must name a predefined memory space
  !$omp target uses_allocators(memspace(renamed_space): my_alloc)
  x = 4
  !$omp end target
end subroutine

subroutine uses_allocators_predefined_shadow
  use omp_lib, only: omp_allocator_handle_kind
  ! A local named constant of the correct handle kind that shadows the spelling
  ! of a predefined allocator. [5.2:182] asks whether the allocator *is* a
  ! predefined allocator, which this is not, so the variable rule applies.
  ! [6.0:315] matches the name instead and accepts it; see
  ! uses-allocators-version60.f90.
  integer(omp_allocator_handle_kind), parameter :: omp_const_mem_alloc = 999
  integer :: x

  !ERROR: A non-predefined allocator 'omp_const_mem_alloc' in a USES_ALLOCATORS clause must be a variable
  !$omp target uses_allocators(omp_const_mem_alloc)
  x = 1
  !$omp end target
end subroutine

subroutine uses_allocators_repeated_clauses
  use omp_lib
  ! As above, each case uses its own allocator.
  integer(omp_allocator_handle_kind) :: map_alloc, priv_alloc, ir_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x, y

  ! The conflict is in the second MAP clause, which is invisible to a scan
  ! that inspects only the first occurrence of each clause id.
  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the MAP clause on the same construct
  !$omp target map(to: y) map(tofrom: map_alloc) uses_allocators(traits(tr): map_alloc)
  x = 1
  !$omp end target

  ! The same, for a repeated PRIVATE clause.
  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the PRIVATE clause on the same construct
  !ERROR: Variable 'y' may not appear on both MAP and PRIVATE clauses on a TARGET construct
  !$omp target private(y) private(priv_alloc) uses_allocators(traits(tr): priv_alloc)
  x = 2
  !$omp end target

  ! IN_REDUCTION is a data-sharing attribute clause allowed on TARGET, but it
  ! is not one of the privatizing clauses.
  !$omp taskgroup task_reduction(+: ir_alloc)
  !ERROR: An allocator in a USES_ALLOCATORS clause cannot also appear in the IN_REDUCTION clause on the same construct
  !$omp target in_reduction(+: ir_alloc) uses_allocators(traits(tr): ir_alloc)
  x = 3
  !$omp end target
  !$omp end taskgroup
end subroutine
