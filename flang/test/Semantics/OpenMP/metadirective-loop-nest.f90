!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine collapse_too_deep(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine ordered_too_deep(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: ORDERED clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do ordered(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine collapse_too_deep_exec(n, a)
  integer :: n, a(n, n), i, j
  a = 0
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(3)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine collapse_too_deep_compiler_directive(n, a)
  integer :: n, a(n, n), i, j
  a = 0
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(3)) default(nothing)
  !dir$ ivdep
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine noncanonical_do_while(n)
  integer :: n, i
  i = 0
  !ERROR: This construct requires a canonical loop nest
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  !BECAUSE: DO WHILE loop is not a valid affected loop
  do while (i < n)
    i = i + 1
  end do
end subroutine

subroutine noncanonical_do_concurrent(n, a)
  integer :: n, a(n), i
  !ERROR: This construct requires a canonical loop nest
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  !BECAUSE: DO CONCURRENT loop is not a valid affected loop
  do concurrent(i=1:n)
    a(i) = i
  end do
end subroutine

subroutine noncanonical_no_control(n)
  integer :: n, i
  i = 0
  !ERROR: This construct requires a canonical loop nest
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  !BECAUSE: DO loop without loop control is not a valid affected loop
  do
    i = i + 1
    if (i >= n) exit
  end do
end subroutine

subroutine collapse_too_deep_interface(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a nest of depth 3, but the associated nest is a nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(3)) default(nothing)
  interface
    subroutine ext()
    end subroutine
  end interface
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine tile_non_rectangular(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: None of the loops affected by TILE can be non-rectangular
  !$omp metadirective when(implementation={vendor(llvm)}: tile sizes(2, 2)) default(nothing)
  do i = 1, n
    !BECAUSE: The upper bound of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 1, i
      a(j, i) = i
    end do
  end do
end subroutine

subroutine collapse_valid(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(2)) default(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i
    end do
  end do
end subroutine

subroutine collapse_non_rectangular_valid(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(2)) default(nothing)
  do i = 1, n
    do j = 1, i
      a(j, i) = i
    end do
  end do
end subroutine

! A loop-associated variant with no loop nest to associate with is in error,
! whether the metadirective is the last construct in the execution part ...
subroutine no_loop_at_end()
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
end subroutine

! ... or is followed by a non-loop construct.
subroutine no_loop_before_stmt(a)
  integer :: a
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: parallel do) default(nothing)
  a = 0
end subroutine

! A variant that cannot be selected on this target needs no loop nest.
subroutine no_loop_dead_variant()
  !$omp metadirective when(device={kind(nohost)}: do) default(nothing)
end subroutine

! A loop-associated variant in a declarative context (e.g. a module
! specification part) also has no loop nest to associate with.
module no_loop_in_module
  implicit none
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
end module

! Starting a contained procedure must not discard a pending variant from the
! enclosing module specification part.
module no_loop_in_module_with_contains
  implicit none
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
contains
  subroutine contained()
  end subroutine
end module

! A later specification-part metadirective must not discard an earlier one.
module no_loop_before_another_metadirective
  implicit none
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  !$omp metadirective when(implementation={vendor(llvm)}: parallel) default(nothing)
end module

! A loop in the enclosing subprogram cannot satisfy a variant from an
! interface body, which has no execution part of its own.
subroutine no_loop_in_interface_body(n, a)
  integer :: n, a(n), i
  interface
    subroutine iface()
      !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
      !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
    end subroutine
  end interface
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Checking an interface body must preserve variants pending in the enclosing
! subprogram for its execution part.
subroutine no_loop_in_interface_body_preserves_outer(n, a)
  integer :: n, a(n), i
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp metadirective when(implementation={vendor(llvm)}: do collapse(2)) default(nothing)
  interface
    subroutine iface()
      !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
      !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
    end subroutine
  end interface
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A loop after an empty BLOCK cannot satisfy a variant from the BLOCK's
! specification part.
subroutine no_loop_in_empty_block(n, a)
  integer :: n, a(n), i
  block
    !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
    !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  end block
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A loop in the BLOCK itself remains its associated loop.
subroutine loop_in_block(n, a)
  integer :: n, a(n), i
  block
    !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
    do i = 1, n
      a(i) = i
    end do
  end block
end subroutine

! A loop in a mutually exclusive IF branch cannot satisfy the variant.
subroutine no_loop_across_if_branch(n, a, flag)
  integer :: n, a(n), i
  logical :: flag
  if (flag) then
    !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
    !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  else
    do i = 1, n
      a(i) = i
    end do
  end if
end subroutine

! A loop after a nested executable block cannot satisfy its variant.
subroutine no_loop_after_if(n, a, flag)
  integer :: n, a(n), i
  logical :: flag
  if (flag) then
    !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
    !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
  end if
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A loop in the same IF branch remains the associated loop.
subroutine loop_in_if_branch(n, a, flag)
  integer :: n, a(n), i
  logical :: flag
  if (flag) then
    !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
    do i = 1, n
      a(i) = i
    end do
  end if
end subroutine
