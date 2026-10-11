! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-hlfir %s -o - | FileCheck %s \
! RUN:   --implicit-check-not='fir.call @_QMsource_contextPbound_arch' %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=60 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-hlfir %s -o - | FileCheck %s \
! RUN:   --implicit-check-not='fir.call @_QMsource_contextPbound_arch' %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-fir %s -o - | FileCheck %s \
! RUN:   --implicit-check-not='fir.call @_QMsource_contextPbound_arch' %}

! DECLARE VARIANT and METADIRECTIVE use the same source construct context.

module source_context
contains
  subroutine depth_base
    !$omp declare variant(depth_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(depth_arch) match(device={arch(x86_64)})
  end subroutine
  subroutine depth_vendor
  end subroutine
  subroutine depth_arch
  end subroutine

  subroutine order_base
    !$omp declare variant(order_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(order_parallel) match(construct={parallel})
  end subroutine
  subroutine order_vendor
  end subroutine
  subroutine order_parallel
  end subroutine

  integer function value_base()
    !$omp declare variant(value_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(value_arch) match(device={arch(x86_64)})
    value_base = 0
  end function
  integer function value_vendor()
    value_vendor = 1
  end function
  integer function value_arch()
    value_arch = 2
  end function

  integer function bound_base(n)
    integer, intent(in) :: n
    !$omp declare variant(bound_vendor) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    !$omp declare variant(bound_arch) match(device={arch(x86_64)})
    bound_base = n
  end function
  integer function bound_vendor(n)
    integer, intent(in) :: n
    bound_vendor = n
  end function
  integer function bound_arch(n)
    integer, intent(in) :: n
    bound_arch = n
  end function

! Bounds exclude TILE in both evaluations; the body includes it.
! CHECK-LABEL: func.func @_QMsource_contextPtile_context(
! CHECK-COUNT-2: fir.call @_QMsource_contextPbound_vendor(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine tile_context(n)
    integer :: n, i
    !$omp tile sizes(2)
    do i = 1, bound_base(n)
      call depth_base()
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPunroll_context(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine unroll_context(n)
    integer :: n, i
    !$omp unroll partial(2)
    do i = 1, n
      call depth_base()
    end do
  end subroutine

! Bounds exclude FUSE in both evaluations of each loop; the bodies include it.
! CHECK-LABEL: func.func @_QMsource_contextPfuse_context(
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK-COUNT-2: fir.call @_QMsource_contextPbound_vendor(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-COUNT-2: fir.call @_QMsource_contextPbound_vendor(
! CHECK: fir.call @_QMsource_contextPdepth_arch()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_vendor
! CHECK: return
  subroutine fuse_context(n)
    integer :: n, i, j
    !$omp fuse
    do i = 1, bound_base(n)
      call depth_base()
    end do
    do j = 1, bound_base(n)
      call depth_base()
    end do
    !$omp end fuse
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPatomic_context(
! CHECK: fir.call @_QMsource_contextPvalue_arch()
! CHECK-NOT: fir.call @_QMsource_contextPvalue_vendor
! CHECK: omp.atomic.update
! CHECK: return
  subroutine atomic_context(x)
    integer :: x
    !$omp atomic update
    x = x + value_base()
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPassume_context()
! CHECK: fir.call @_QMsource_contextPdepth_vendor()
! CHECK-NOT: fir.call @_QMsource_contextPdepth_arch
! CHECK: return
  subroutine assume_context
    !$omp assume holds(.true.)
      call depth_base()
    !$omp end assume
  end subroutine

! CHECK-LABEL: func.func @_QMsource_contextPcombined_context(
! CHECK: fir.call @_QMsource_contextPorder_parallel()
! CHECK-NOT: fir.call @_QMsource_contextPorder_vendor
! CHECK: return
  subroutine combined_context(n)
    integer :: n, i
    !$omp teams distribute parallel do
    do i = 1, n
      call order_base()
    end do
  end subroutine
end module

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
! CHECK: omp.target
! CHECK: omp.barrier
! CHECK: omp.single
! CHECK: omp.taskyield
! CHECK: omp.terminator
! CHECK: omp.terminator
! CHECK-NOT: omp.target
! CHECK-NOT: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine standalone_block()
  call before_block()
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    !$omp metadirective when(construct={target}: barrier)
    !$omp single
      !$omp metadirective &
      !$omp& when(construct={target, single}: taskyield)
    !$omp end single
  end block
  !$omp metadirective when(construct={target}: taskyield)
end subroutine

! Each runtime alternative lowers the BLOCK once, with its own context.
! CHECK-LABEL: func.func @_QPruntime_block(
! CHECK: fir.if
! CHECK: omp.target
! CHECK: fir.call @_QPuse_block_local
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK: omp.taskyield
! CHECK: omp.terminator
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK: } else {
! CHECK-NOT: omp.target
! CHECK-NOT: omp.taskyield
! CHECK: fir.call @_QPuse_block_local
! CHECK-NOT: fir.call @_QPuse_block_local
! CHECK-NOT: omp.target
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine runtime_block(flag)
  logical :: flag
  !$omp metadirective when(user={condition(flag)}: target) &
  !$omp& otherwise(nothing)
  block
    integer :: local
    local = 1
    call use_block_local(local)
    !$omp metadirective when(construct={target}: taskyield)
  end block
  !$omp metadirective when(construct={target}: taskyield)
end subroutine

! An ignored compiler directive does not separate the BLOCK from its
! replacement.
! CHECK-LABEL: func.func @_QPstandalone_block_ignored_directive(
! CHECK: omp.target
! CHECK: omp.taskyield
! CHECK: omp.terminator
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine standalone_block_ignored_directive(a)
  integer :: a(10)
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  !dir$ ignored_comment
  !dir$ ignored(1)
  !dir$ loop count(10)
  !dir$ assume_aligned a:64
  block
    !$omp metadirective when(construct={target}: taskyield) otherwise(taskwait)
  end block
end subroutine

! A compiler directive with an effect still ends the search for the BLOCK.
! CHECK-LABEL: func.func @_QPstandalone_block_prefetch(
! CHECK: omp.target
! CHECK-NOT: fir.prefetch
! CHECK: omp.terminator
! CHECK: fir.prefetch
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK: return
subroutine standalone_block_prefetch(a)
  integer :: a(10)
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  !dir$ prefetch a
  block
    !$omp metadirective when(construct={target}: taskyield) otherwise(taskwait)
  end block
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

! The selected TARGET makes the inner SIMD variant lose to NOTHING.
! CHECK-LABEL: func.func @_QPstandalone_block_loop(
! CHECK: omp.target
! CHECK-NOT: omp.simd
! CHECK: fir.do_loop
! CHECK: omp.terminator
! CHECK-NOT: fir.do_loop
! CHECK-NOT: omp.simd
! CHECK: return
subroutine standalone_block_loop(n)
  integer :: n, i
  !$omp metadirective when(implementation={vendor(llvm)}: target) &
  !$omp& otherwise(nothing)
  block
    !$omp metadirective &
    !$omp& when(construct={target}: nothing) &
    !$omp& when(implementation={vendor(score(0): llvm)}: simd collapse(1)) &
    !$omp& otherwise(nothing)
    do i = 1, n
    end do
  end block
end subroutine
