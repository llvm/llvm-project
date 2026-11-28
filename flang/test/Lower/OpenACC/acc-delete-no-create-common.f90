! Test lowering of a named common block in the DELETE clause of EXIT DATA
! and in the NO_CREATE clause.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine exit_data_delete_common()
  integer :: a
  common /c1/ a
  !$acc enter data create(/c1/)
  a = 1
  !$acc exit data delete(/c1/)
end subroutine

! CHECK-LABEL: func.func @_QPexit_data_delete_common()
! CHECK:         %[[C1:.*]] = fir.address_of(@c1_) : !fir.ref<!fir.array<4xi8>>
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[C1]] : !fir.ref<!fir.array<4xi8>>) dataClause(acc_delete) structured(false) name("c1")
! CHECK:         acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<4xi8>>)
! CHECK:         acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<4xi8>>) structured(false) name("c1")

subroutine data_no_create_common()
  integer :: b
  common /c2/ b
  !$acc data no_create(/c2/)
  b = 1
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPdata_no_create_common()
! CHECK:         %[[C2:.*]] = fir.address_of(@c2_) : !fir.ref<!fir.array<4xi8>>
! CHECK:         %[[NOCREATE:.*]] = acc.nocreate varPtr(%[[C2]] : !fir.ref<!fir.array<4xi8>>) name("c2")
! CHECK:         acc.data dataOperands(%[[NOCREATE]] : !fir.ref<!fir.array<4xi8>>)
! CHECK:         acc.delete accPtr(%[[NOCREATE]] : !fir.ref<!fir.array<4xi8>>) dataClause(acc_no_create) name("c2")
