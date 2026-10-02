! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-hlfir %s -o - | FileCheck %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-fir %s -o - | FileCheck %s %}

! Executable loop transformations contribute to the construct context.
! CHECK-LABEL: func.func @_QPtile_context(
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine tile_context(n, a)
  integer :: n, a(n, n), i, j
  !$omp tile sizes(2)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
    do j = 1, n
      a(j, i) = j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_context(
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine unroll_context(n, a)
  integer :: n, a(n), i
  !$omp unroll partial(2)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
    a(i) = i
  end do
end subroutine

! Informational directives do not contribute to the construct context.
! CHECK-LABEL: func.func @_QPassume_context()
! CHECK: omp.taskyield
! CHECK-NOT: omp.barrier
! CHECK: return
subroutine assume_context()
  !$omp assume holds(.true.)
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
  !$omp end assume
end subroutine

! Combined and composite constructs retain their source nesting order.
! CHECK-LABEL: func.func @_QPcombined_context(
! CHECK: omp.teams
! CHECK: omp.parallel
! CHECK: omp.distribute
! CHECK: omp.wsloop
! CHECK: omp.taskyield
! CHECK-NOT: omp.barrier
! CHECK: return
subroutine combined_context(n)
  integer :: n, i
  !$omp teams distribute parallel do
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: barrier) &
    !$omp& when(construct={parallel}: taskyield)
  end do
end subroutine

! A standalone replacement owns the following BLOCK and its nested context.
! CHECK-LABEL: func.func @_QPstandalone_block()
! CHECK: fir.call @_QPbefore_block
! CHECK: omp.parallel
! CHECK: omp.barrier
! CHECK: omp.single
! CHECK: omp.taskyield
! CHECK: omp.terminator
! CHECK: omp.terminator
! CHECK-NOT: omp.parallel
! CHECK-NOT: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine standalone_block()
  call before_block()
  !$omp metadirective when(implementation={vendor(llvm)}: parallel)
  block
    !$omp metadirective when(construct={parallel}: barrier)
    !$omp single
      !$omp metadirective &
      !$omp& when(construct={parallel, single}: taskyield)
    !$omp end single
  end block
  !$omp metadirective when(construct={parallel}: taskyield)
end subroutine

! Each runtime alternative lowers the BLOCK once, with its own context.
! CHECK-LABEL: func.func @_QPruntime_block(
! CHECK: fir.if
! CHECK: omp.parallel
! CHECK: fir.call @_QPuse_block_local
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK: omp.taskyield
! CHECK: omp.terminator
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK: } else {
! CHECK-NOT: omp.parallel
! CHECK-NOT: omp.taskyield
! CHECK: fir.call @_QPuse_block_local
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK-NOT: omp.parallel
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine runtime_block(flag)
  logical :: flag
  !$omp metadirective when(user={condition(flag)}: parallel) &
  !$omp& otherwise(nothing)
  block
    integer :: local
    local = 1
    call use_block_local(local)
    !$omp metadirective when(construct={parallel}: taskyield)
  end block
  !$omp metadirective when(construct={parallel}: taskyield)
end subroutine

! An empty delimited replacement must not capture the following BLOCK.
! CHECK-LABEL: func.func @_QPempty_delimited_block()
! CHECK-NOT: omp.parallel
! CHECK: fir.call @_QPafter_delimited
! CHECK-NOT: omp.taskyield
! CHECK-NOT: fir.call @_QPafter_delimited
! CHECK: return
subroutine empty_delimited_block()
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel)
  !$omp end metadirective
  block
    call after_delimited()
    !$omp metadirective when(construct={parallel}: taskyield)
  end block
end subroutine

! The selected PARALLEL makes the inner SIMD variant lose to NOTHING.
! CHECK-LABEL: func.func @_QPstandalone_block_loop(
! CHECK: omp.parallel
! CHECK-NOT: omp.simd
! CHECK: fir.do_loop
! CHECK: omp.terminator
! CHECK-NOT: fir.do_loop
! CHECK-NOT: omp.simd
! CHECK: return
subroutine standalone_block_loop(n)
  integer :: n, i
  !$omp metadirective when(implementation={vendor(llvm)}: parallel) &
  !$omp& otherwise(nothing)
  block
    !$omp metadirective &
    !$omp& when(construct={parallel}: nothing) &
    !$omp& when(implementation={vendor(score(0): llvm)}: simd collapse(1)) &
    !$omp& otherwise(nothing)
    do i = 1, n
    end do
  end block
end subroutine
