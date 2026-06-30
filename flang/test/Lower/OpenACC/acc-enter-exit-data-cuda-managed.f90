! Test that OpenACC enter/exit data clauses are skipped for explicit CUDA
! managed objects, while non-managed objects in the same clause are
! still lowered.
!
! RUN: bbc -fcuda -fopenacc -emit-hlfir -gpu=managed %s -o - | FileCheck %s

subroutine managed_only()
  real, allocatable :: m(:)
  allocate(m(10))
  !$acc enter data copyin(m)
  !$acc update device(m)
  !$acc exit data copyout(m)
end subroutine

! CHECK-LABEL: func.func @_QPmanaged_only()
! CHECK-NOT: acc.copyin
! CHECK-NOT: acc.enter_data
! CHECK-NOT: acc.update
! CHECK-NOT: acc.exit_data
! CHECK-NOT: acc.copyout

subroutine mixed_managed_and_regular()
  real, allocatable :: m(:)
  real :: h(10)
  !$acc enter data copyin(m) copyin(h)
  !$acc exit data copyout(m) copyout(h)
end subroutine

! CHECK-LABEL: func.func @_QPmixed_managed_and_regular()
! CHECK: %[[H_COPYIN:.*]] = acc.copyin
! CHECK-SAME: {name = "h", structured = false}
! CHECK-NOT: {name = "m", structured = false}
! CHECK: acc.enter_data dataOperands(%[[H_COPYIN]]
! CHECK: %[[H_DEVPTR:.*]] = acc.getdeviceptr
! CHECK-SAME: {dataClause = #acc<data_clause acc_copyout>, name = "h", structured = false}
! CHECK-NOT: {dataClause = #acc<data_clause acc_copyout>, name = "m", structured = false}
! CHECK: acc.exit_data dataOperands(%[[H_DEVPTR]]
! CHECK: acc.copyout accPtr(%[[H_DEVPTR]]
! CHECK-SAME: {name = "h", structured = false}
