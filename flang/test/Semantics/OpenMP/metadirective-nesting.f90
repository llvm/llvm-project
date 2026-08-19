! RUN: split-file %s %t
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only \
! RUN:   %t/invalid.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only \
! RUN:   %t/valid.f90

! CHECK: error: A worksharing region may not be closely nested
! CHECK: error: A worksharing region may not be closely nested
! CHECK: error: A worksharing region may not be closely nested
! CHECK: error: A worksharing region may not be closely nested
! CHECK: warning: OpenMP directive MASTER has been deprecated
! CHECK: error: `MASTER` region may not be closely nested
! CHECK: error: An ORDERED directive without the DEPEND clause
! CHECK: error: The only OpenMP constructs that can be encountered
! CHECK: error: With DO clause, CANCEL construct cannot be closely nested
! CHECK: error: An ORDERED construct with the DEPEND clause
! CHECK: error: CANCEL DO directive is not closely nested
! CHECK: error: The CANCEL construct cannot be nested inside
! CHECK: error: The CANCEL construct cannot be nested inside
! CHECK: error: The CANCEL construct cannot be nested inside
! CHECK: error: The number of variables in the SINK iteration vector

!--- invalid.f90
subroutine selected_do_single(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp single
    print *, i
    !$omp end single
  end do
end

subroutine selected_do_selected_single(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(construct={do}: single) otherwise(nothing)
    print *, i
    !$omp end metadirective
  end do
end

subroutine single_selected_do(n)
  integer :: n, i
  !$omp single
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    print *, i
  end do
  !$omp end single
end

subroutine selected_do_sections(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp sections
    !$omp section
    print *, i
    !$omp end sections
  end do
end

subroutine selected_do_master(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp master
    print *, i
    !$omp end master
  end do
end

subroutine selected_do_ordered(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp ordered
    print *, i
    !$omp end ordered
  end do
end

subroutine selected_simd_single(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: simd) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp single
    print *, i
    !$omp end single
  end do
end

subroutine dynamic_do_cancel(flag, n)
  logical :: flag
  integer :: n, i
  !$omp parallel
  !$omp metadirective when(user={condition(flag)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp cancel do
  end do
  !$omp end parallel
end

subroutine selected_ordered_depend(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={do}: ordered depend(source)) &
    !$omp& otherwise(nothing)
  end do
end

subroutine orphan_selected_cancel()
  !$omp metadirective when(user={condition(.true.)}: cancel do) &
  !$omp& otherwise(nothing)
  print *, "not a loop"
end

subroutine cancel_selected_do_ordered(n)
  integer :: n, i
  !$omp parallel
  !$omp metadirective when(user={condition(.true.)}: do ordered) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp cancel do
  end do
  !$omp end parallel
end

subroutine selected_cancel_selected_do_ordered(n)
  integer :: n, i
  !$omp parallel
  !$omp metadirective when(user={condition(.true.)}: do ordered) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective when(construct={do}: cancel do) &
    !$omp& otherwise(nothing)
  end do
  !$omp end parallel
end

subroutine selected_cancel_end_do_nowait(n)
  integer :: n, i
  !$omp parallel
  !$omp do
  do i = 1, n
    !$omp metadirective when(construct={do}: cancel do) &
    !$omp& otherwise(nothing)
  end do
  !$omp end do nowait
  !$omp end parallel
end

subroutine selected_ordered_sink(n)
  integer :: n, i, j
  !$omp metadirective when(user={condition(.true.)}: do ordered(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    do j = 1, n
      !$omp metadirective &
      !$omp& when(construct={do}: ordered depend(sink: i - 1)) &
      !$omp& otherwise(nothing)
    end do
  end do
end

!--- valid.f90
subroutine selected_do_cancel(n)
  integer :: n, i
  !$omp parallel
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp cancel do
  end do
  !$omp end parallel
end

subroutine parallel_breaks_close_nesting(n)
  integer :: n, i
  !$omp metadirective when(user={condition(.true.)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp parallel
    !$omp single
    print *, i
    !$omp end single
    !$omp end parallel
  end do
end

subroutine correlated_replacements(flag, n)
  logical :: flag
  integer :: n, i
  !$omp metadirective when(user={condition(flag)}: do) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective when(construct={do}: nothing) &
    !$omp& otherwise(barrier)
  end do
end

subroutine nested_selected_simd(n, m)
  integer :: n, m, i, j
  !$omp metadirective when(user={condition(.true.)}: simd) &
  !$omp& otherwise(nothing)
  do i = 1, n
    !$omp metadirective when(construct={simd}: simd) &
    !$omp& otherwise(nothing)
    do j = 1, m
      print *, i, j
    end do
  end do
end

subroutine scan_in_selected_simd(a, n)
  integer :: n, i, sum
  integer :: a(n)
  sum = 0
  !$omp metadirective &
  !$omp& when(user={condition(.true.)}: simd reduction(inscan, +:sum)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    sum = sum + a(i)
    !$omp scan inclusive(sum)
    a(i) = sum
  end do
end
