! A directive construct is the loop it owns, so a loop whose branching is
! confined to its body carries the construct holding it along when it is
! reclassified. The loop keeps its bounds and the body's branching is folded
! into a region inside it.

! RUN: %flang_fc1 -fopenmp -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s | FileCheck %s --check-prefix=FIR

subroutine wsloop_cycle(a, n)
  real :: a(n)
  integer :: n, i
  !$omp parallel do
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      cycle
    end if
    a(i) = 2.0
  end do
  !$omp end parallel do
end subroutine

! CHECK: Subroutine wsloop_cycle
! CHECK: <<OpenMPConstruct~>>
! CHECK: <<DoConstruct~>>

! FIR-LABEL: func.func @_QPwsloop_cycle
! FIR:         omp.parallel {
! FIR:           omp.wsloop private({{.*}}) {
! FIR:             omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! FIR:               scf.execute_region no_inline {
! FIR:                 cf.cond_br
! FIR:                 scf.yield
! FIR:               }
