!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine collapse_too_deep(n, a)
  integer :: n, a(n, n), i, j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
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
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
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
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
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
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
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
