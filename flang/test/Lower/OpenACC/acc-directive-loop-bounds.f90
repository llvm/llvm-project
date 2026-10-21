! A directive construct is the loop it owns: its lowering reads the construct's
! own classification to decide whether the loop op carries its bounds. A loop
! whose branching is confined to its body keeps its structured form, so the
! construct holding it must be reclassified too -- otherwise the directive gets
! a bounds-free acc.loop that nothing can partition, with the real loop nested
! inside it.

! RUN: %flang_fc1 -fopenacc -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenacc -emit-hlfir -o - %s | FileCheck %s --check-prefix=FIR

! The CYCLE keeps its branching inside the loop body, so both the loop and the
! construct that owns it are reclassified.
subroutine parallel_loop_cycle(a, n)
  real :: a(n)
  integer :: n, i
  !$acc parallel loop
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      cycle
    end if
    a(i) = 2.0
  end do
end subroutine

! CHECK: Subroutine parallel_loop_cycle
! CHECK: <<OpenACCConstruct~>>
! CHECK: <<DoConstruct~>>

! One acc.loop, and it is the directive's own: it carries the bounds, with the
! body's branching folded into a region inside it. A second, nested acc.loop
! here would mean the bounds landed on a loop the directive does not own.
! FIR-LABEL: func.func @_QPparallel_loop_cycle
! FIR:         acc.parallel combined(loop) {
! FIR-NOT:       acc.loop
! FIR:           acc.loop combined(parallel) private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! FIR:             scf.execute_region no_inline {
! FIR:               cf.cond_br
! FIR:               scf.yield
! FIR:             }
! FIR-NOT:       acc.loop
! FIR:           acc.yield
! FIR:         }

! Negative: the construct holds a GOTO of its own, so its branching is not
! confined to the loop and it stays unstructured. The rewrite of a single-
! statement IF body does not reach this one, so the GOTO survives.
subroutine parallel_region_goto(a, n)
  real :: a(n)
  integer :: n, i
  !$acc parallel
  if (n > 0) then
    a(1) = 0.0
    goto 90
  end if
  !$acc loop
  do i = 1, n
    a(i) = 2.0
  end do
90 continue
  !$acc end parallel
end subroutine

! CHECK: Subroutine parallel_region_goto
! CHECK: <<OpenACCConstruct!>>
! CHECK: GotoStmt!
