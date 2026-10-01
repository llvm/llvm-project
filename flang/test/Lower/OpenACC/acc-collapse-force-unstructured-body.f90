! collapse(force:2) over a nest whose inner body branches within itself.
!
! The force lowering absorbs both levels into the directive's own acc.loop and
! sinks whatever sits between them, so the bounds of both levels belong on that
! op. The branching stays inside the body, which therefore still needs folding
! into a region.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - | FileCheck %s

! A CYCLE in the inner body: it branches to that body's own end.
subroutine collapse_force_cycle(a, n, m)
  integer :: n, m, i, j
  real :: a(n,m)

  !$acc parallel loop collapse(force:2)
  do i = 1, n
    do j = 1, m
      if (a(i,j) > 0.0) then
        a(i,j) = 1.0
        cycle
      end if
      a(i,j) = 2.0
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_cycle
! CHECK:         acc.parallel combined(loop)
! CHECK:           acc.loop combined(parallel) private({{.*}}) control(%{{.*}} : i32, %{{.*}} : i32) = (%{{.*}}, %{{.*}} : i32, i32) to (%{{.*}}, %{{.*}} : i32, i32) step (%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:             scf.execute_region no_inline {
! CHECK-NOT:         acc.loop
! CHECK:           acc.yield
! CHECK-NEXT:    } inclusiveUpperbound({{.*}}) collapse([2])

! The same nest with GOTOs to labels later in the inner body. Two of them, to
! two targets: a single IF-guarded GOTO over one statement is rewritten into a
! fir.if and never reaches the raw form this exercises.
subroutine collapse_force_goto(a, n, m)
  integer :: n, m, i, j
  real :: a(n,m)

  !$acc parallel loop collapse(force:2)
  do i = 1, n
    do j = 1, m
      if (a(i,j) > 0.0) goto 20
      if (a(i,j) < -1.0) goto 30
      a(i,j) = 1.0
      goto 40
20    a(i,j) = 2.0
      goto 40
30    a(i,j) = 3.0
40    continue
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_goto
! CHECK:         acc.parallel combined(loop)
! CHECK:           acc.loop combined(parallel) private({{.*}}) control(%{{.*}} : i32, %{{.*}} : i32) = (%{{.*}}, %{{.*}} : i32, i32) to (%{{.*}}, %{{.*}} : i32, i32) step (%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:             scf.execute_region no_inline {
! CHECK:               cf.cond_br
! CHECK-NOT:         acc.loop
! CHECK:           acc.yield
! CHECK-NEXT:    } inclusiveUpperbound({{.*}}) collapse([2])
