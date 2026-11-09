! An unstructured-CFG pattern inside a combined `acc parallel loop` construct
! (default parallelism is `independent`).
!
! The IF-guarded CYCLE branches only within the loop body, so the loop keeps
! its structured form and carries the construct that owns it along. The bounds
! of both collapsed levels land on the directive's own acc.loop, whichever way
! --emit-independent-loops-as-unstructured is set.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - | FileCheck %s

subroutine test_unstructured_collapse_cycle(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc parallel loop collapse(2) copy(a)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_collapse_cycle
! CHECK:         acc.parallel combined(loop)
! A second acc.loop nested here would mean the bounds landed on a loop the
! directive does not own.
! CHECK-NOT:       acc.loop
! CHECK:           acc.loop combined(parallel) private({{.*}}) control(%{{.*}} : i32, %{{.*}} : i32) = (%{{.*}}, %{{.*}} : i32, i32) to (%{{.*}}, %{{.*}} : i32, i32) step (%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:             scf.execute_region no_inline {
! CHECK:               cf.cond_br
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound({{.*}}) collapse([2]) collapseDeviceType({{.*}}) independent
!
! Nothing else is nested in the compute region, which closes structured.
! CHECK-NOT:       acc.loop
! CHECK:         acc.yield
! CHECK-NEXT:    }
