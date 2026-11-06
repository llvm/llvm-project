! Test a COMMON block declared in a module and named in a data clause of a
! module procedure that does not otherwise reference any of its members.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_module_common
  integer :: a, b
  common /c/ a, b
contains
  subroutine enter_data_create()
    !$acc enter data create(/c/)
  end subroutine
  subroutine exit_data_copyout()
    !$acc exit data copyout(/c/)
  end subroutine
  subroutine update_device()
    !$acc update device(/c/)
  end subroutine
  subroutine data_present()
    !$acc data present(/c/)
    !$acc end data
  end subroutine
end module

! CHECK-LABEL: func.func @_QMacc_module_commonPenter_data_create()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")
! CHECK:         acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_commonPexit_data_copyout()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) dataClause(acc_copyout) structured(false) name("c")
! CHECK:         acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<8xi8>>)
! CHECK:         acc.copyout accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<8xi8>>) to varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")

! CHECK-LABEL: func.func @_QMacc_module_commonPupdate_device()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[UPD:.*]] = acc.update_device varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")
! CHECK:         acc.update dataOperands(%[[UPD]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_commonPdata_present()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[PRESENT:.*]] = acc.present varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) name("c")
! CHECK:         acc.data dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<8xi8>>)
