! RUN: bbc -fcuda -fopenacc -emit-hlfir -gpu=managed %s -o - | FileCheck %s

module acc_declare_managed_no_global_ctor
  integer, allocatable :: data(:)
  !$acc declare create(data)

  integer, allocatable :: data2(:)
  !$acc declare copyin(data2)
contains
  subroutine init()
    allocate(data(16))
    allocate(data2(16))
  end subroutine
end module

! CHECK-LABEL: func.func @_QMacc_declare_managed_no_global_ctorPinit()
! CHECK-COUNT-2: cuf.allocate

! CHECK-LABEL: func.func @_QMacc_declare_managed_no_global_ctorEdata_acc_declare_post_alloc() attributes {acc.declare_action} {
! CHECK: %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_managed_no_global_ctorEdata)
! CHECK: %[[CREATE_DESC:.*]] = acc.create varPtr(%[[GLOBAL_ADDR]]
! CHECK: acc.declare_enter dataOperands(%[[CREATE_DESC]]

! CHECK-LABEL: func.func @_QMacc_declare_managed_no_global_ctorEdata2_acc_declare_post_alloc() attributes {acc.declare_action} {
! CHECK: %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_managed_no_global_ctorEdata2)
! CHECK: %[[COPYIN_DESC:.*]] = acc.copyin varPtr(%[[GLOBAL_ADDR]]
! CHECK: acc.declare_enter dataOperands(%[[COPYIN_DESC]]

! CHECK-NOT: acc.global_ctor
! CHECK-NOT: acc.global_dtor
