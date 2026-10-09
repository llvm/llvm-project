! A combined `acc parallel loop` whose default parallelism resolves to
! `independent` and whose body branches backwards.
!
! The GOTO jumps back to an earlier statement of the body, so control can reach
! the GOTO again from its own target. The IF it sits in falls through to the end
! of the body, so the cycle can be left: the branching stays confined to the
! body, the loop keeps its structured form, and the directive takes it over.
! Its bounds land on the directive's own acc.loop, whichever way
! --emit-independent-loops-as-unstructured is set.
!
! A GOTO leaving the loop would not serve here: it is rejected on both paths,
! as is EXIT or RETURN in a combined construct.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - | FileCheck %s

subroutine test_combined_backward_goto(a, n)
  integer :: n, i
  real :: a(n)

  !$acc parallel loop
  do i = 1, n
20  continue
    a(i) = a(i) * 2.0
    if (a(i) < 100.0) goto 20
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_combined_backward_goto
! CHECK:         acc.parallel combined(loop)
! A second acc.loop nested here would mean the bounds landed on a loop the
! directive does not own.
! CHECK-NOT:       acc.loop
! CHECK:           acc.loop combined(parallel) private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! The backward branch stays inside the region holding the body.
! CHECK:             scf.execute_region no_inline {
! CHECK:             ^[[TGT:bb[0-9]+]]:  // 2 preds
! CHECK:               cf.cond_br %{{.*}}, ^[[GOTO:bb[0-9]+]], ^[[REST:bb[0-9]+]]
! CHECK:             ^[[GOTO]]:
! CHECK-NEXT:          cf.br ^[[TGT]]
! CHECK:             ^[[REST]]:
! CHECK-NEXT:          scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } {{.*}}independent
! CHECK-NOT:       unstructured
! CHECK:         acc.yield
