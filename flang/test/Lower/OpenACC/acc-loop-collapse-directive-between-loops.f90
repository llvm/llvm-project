! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o %t 2>&1 | FileCheck %s --check-prefix=WARN

! Verify that a compiler directive (e.g. !DIR$ IVDEP) appearing between the
! levels of a collapsed loop nest does not get mistaken for the next nested
! DO CONSTRUCT. The directive is an extra sibling evaluation between the
! NonLabelDoStmt and the inner DoConstruct; the collapse descent must skip
! over it rather than absorbing it as the loop body.

subroutine collapse2_directive_between_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n)
  integer :: i, j

  !$acc parallel loop collapse(2) copy(a)
  do i = 1, n
!DIR$ IVDEP
    do j = 1, n
      a(j,i) = 1
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse2_directive_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) {{.*}} control(%{{[^ ]+}} : i32, %{{[^ ]+}} : i32) =
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([2])

subroutine collapse3_directive_between_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n,n)
  integer :: i, j, k

  !$acc parallel loop collapse(3) copy(a)
  do i = 1, n
!DIR$ NOVECTOR
    do j = 1, n
      do k = 1, n
        a(k,j,i) = 1
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse3_directive_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) {{.*}} control(%{{[^ ]+}} : i32, %{{[^ ]+}} : i32, %{{[^ ]+}} : i32) =
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([3])

! A second directive between the inner loop levels (as opposed to between
! the outer and first inner loop) exercises every iteration of the descent,
! not just the first.
subroutine collapse3_directive_between_inner_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n,n)
  integer :: i, j, k

  !$acc parallel loop collapse(3) copy(a)
  do i = 1, n
    do j = 1, n
!DIR$ UNROLL(2)
      do k = 1, n
        a(k,j,i) = 1
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse3_directive_between_inner_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) {{.*}} control(%{{[^ ]+}} : i32, %{{[^ ]+}} : i32, %{{[^ ]+}} : i32) =
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([3])

subroutine tile_directive_between_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n)
  integer :: i, j

  !$acc parallel loop tile(2, 2) copy(a)
  do i = 1, n
!DIR$ IVDEP
    do j = 1, n
      a(j,i) = 1
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPtile_directive_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) {{.*}} tile({{.*}} control(%{{[^ ]+}} : i32, %{{[^ ]+}} : i32) =
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield

! collapse(force: ...) explicitly allows intervening code between the loop
! levels; the strict per-level descent (and its "directive ignored" warning)
! must not run for it -- the force lowering sinks the prologue statements
! (and the directive) into the collapsed body instead.
subroutine collapse_force_stmt_between_loops(n, a, s)
  integer, intent(in) :: n
  integer :: a(n,n), s
  integer :: i, j

  !$acc parallel loop collapse(force:2) copy(a)
  do i = 1, n
    s = s + i
!DIR$ IVDEP
    do j = 1, n
      a(j,i) = 1
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_stmt_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) {{.*}} control(%{{[^ ]+}} : i32, %{{[^ ]+}} : i32) =
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([2])

! One warning per skipped directive, each carrying the directive's source
! location, in subroutine order; the force subroutine must not warn.
! WARN: warning: loc("{{.*}}acc-loop-collapse-directive-between-loops.f90":{{[0-9]+}}:{{[0-9]+}}): compiler directive ignored: it appears between loop levels of a collapsed or tiled loop nest
! WARN: warning: loc("{{.*}}acc-loop-collapse-directive-between-loops.f90":{{[0-9]+}}:{{[0-9]+}}): compiler directive ignored: it appears between loop levels of a collapsed or tiled loop nest
! WARN: warning: loc("{{.*}}acc-loop-collapse-directive-between-loops.f90":{{[0-9]+}}:{{[0-9]+}}): compiler directive ignored: it appears between loop levels of a collapsed or tiled loop nest
! WARN: warning: loc("{{.*}}acc-loop-collapse-directive-between-loops.f90":{{[0-9]+}}:{{[0-9]+}}): compiler directive ignored: it appears between loop levels of a collapsed or tiled loop nest
! WARN-NOT: compiler directive ignored
