!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine missing_dsa(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    a(i) = x
  end do
end subroutine

subroutine explicit_dsa(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a, x)) default(nothing)
  do i = 1, n
    a(i) = x
  end do
end subroutine

subroutine common_block_dsa(n)
  integer :: n, i, x
  common /block/ x
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, /block/)) default(nothing)
  do i = 1, n
    x = i
  end do
end subroutine

! A DSA from another variant does not apply to this one.
subroutine variant_dsa(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) &
  !$omp& default(parallel do shared(x))
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    a(i) = x
  end do
end subroutine

! A DSA on a host-associated symbol still belongs only to its own variant.
subroutine host_dsa(x)
  integer :: x
contains
  subroutine inner(n, a)
    integer :: n, a(n), i
    !$omp metadirective &
    !$omp& when(implementation={vendor(llvm)}: &
    !$omp& parallel do default(none) shared(n, a)) &
    !$omp& default(parallel do private(x))
    do i = 1, n
      !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
      a(i) = x
    end do
  end subroutine
end subroutine

! A statically inapplicable variant does not constrain the loop.
subroutine inapplicable_variant(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(device={kind(nohost)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    a(i) = x
  end do
end subroutine

! A run-time condition may select the variant, so its loop is checked.
subroutine dynamic_variant(n, a, x, flag)
  integer :: n, a(n), x, i
  logical :: flag
  !$omp metadirective &
  !$omp& when(user={condition(flag)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    a(i) = x
  end do
end subroutine

! An invalid loop nest does not suppress an independent DEFAULT(NONE) error.
subroutine invalid_depth_and_missing_dsa(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp& parallel do collapse(2) default(none) shared(n, a)) default(nothing)
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    a(i) = x
  end do
end subroutine

! Sequential loop indices and automatic variables declared in the region have
! predetermined data-sharing attributes.
subroutine predetermined_dsa(n, a)
  integer :: n, a(n), i, j
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    do j = 1, n
      block
        integer :: local
        local = i + j
        a(j) = local
      end block
    end do
  end do
end subroutine

subroutine static_local(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    block
      integer, save :: saved
      !ERROR: The DEFAULT(NONE) clause requires that 'saved' must be listed in a data-sharing attribute clause
      saved = i
      a(i) = saved
    end block
  end do
end subroutine

subroutine implicit_saved_local(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    block
      integer :: saved = 1
      !ERROR: The DEFAULT(NONE) clause requires that 'saved' must be listed in a data-sharing attribute clause
      saved = i
      a(i) = saved
    end block
  end do
end subroutine

subroutine unused_static_locals(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    block
      integer, save :: explicit_saved
      integer :: implicit_saved = 1
      integer :: saved, datum
      integer, save :: volatile_saved
      save saved
      data datum /1/
      volatile volatile_saved
      a(i) = i
    end block
  end do
end subroutine

subroutine saved_statement_locals(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    block
      integer :: saved, datum
      save saved
      data datum /1/
      !ERROR: The DEFAULT(NONE) clause requires that 'saved' must be listed in a data-sharing attribute clause
      saved = i
      !ERROR: The DEFAULT(NONE) clause requires that 'datum' must be listed in a data-sharing attribute clause
      a(i) = datum
    end block
  end do
end subroutine

subroutine declaration_bound(n, a, extent)
  integer :: n, a(n), extent, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    block
      !ERROR: The DEFAULT(NONE) clause requires that 'extent' must be listed in a data-sharing attribute clause
      integer :: local(extent)
      a(i) = size(local)
    end block
  end do
end subroutine

subroutine nested_private(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n)) default(nothing)
  do i = 1, n
    !$omp task private(a)
    a(i) = i
    !$omp end task
  end do
end subroutine

subroutine nested_firstprivate(n, a, x)
  integer :: n, a(n), x, i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n, a)) default(nothing)
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    !$omp task firstprivate(x)
    a(i) = x
    !$omp end task
  end do
end subroutine

subroutine nested_implicit(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n)) default(nothing)
  do i = 1, n
    !$omp task
    !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
    a(i) = i
    !$omp end task
  end do
end subroutine

subroutine nested_shared(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none) shared(n)) default(nothing)
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
    !$omp task shared(a)
    a(i) = i
    !$omp end task
  end do
end subroutine
