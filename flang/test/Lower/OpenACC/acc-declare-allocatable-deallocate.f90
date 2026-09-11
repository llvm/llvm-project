! Test !$acc declare create on a module allocatable: the device copy's
! lifetime follows the host allocation, so it must be released before that
! allocation is freed.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt --acc-declare-action-conversion -o - | FileCheck %s --check-prefix=CONV

module acc_declare_dealloc_mod
  integer, allocatable :: data(:)
  !$acc declare create(data)
contains
  subroutine free_data()
    deallocate(data)
  end subroutine
end module

! The deallocation site must reference the pre-dealloc recipe.
! CHECK-LABEL: func.func @_QMacc_declare_dealloc_modPfree_data()
! CHECK:         fir.box_addr
! CHECK-SAME:      acc.declare_action<preDealloc = @_QMacc_declare_dealloc_modEdata_acc_declare_pre_dealloc>

! No post-dealloc recipe is emitted: it would run after the host storage is
! freed, when the descriptor no longer holds the address the mapping uses.
! CHECK-NOT: _acc_declare_post_dealloc

! The release happens here, before the host deallocation.
! CHECK-LABEL: func.func @_QMacc_declare_dealloc_modEdata_acc_declare_pre_dealloc() attributes {acc.declare_action} {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMacc_declare_dealloc_modEdata)
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ADDR]]
! CHECK:         acc.declare_exit dataOperands(%[[DEVPTR]]

! CONV-LABEL: func.func @_QMacc_declare_dealloc_modPfree_data()
! CONV:         fir.call @_QMacc_declare_dealloc_modEdata_acc_declare_pre_dealloc()
