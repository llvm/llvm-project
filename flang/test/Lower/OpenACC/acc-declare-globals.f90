! This test checks lowering of OpenACC declare directive in module specification
! part.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_declare_test
 integer, parameter :: n = 100000
 real, dimension(n) :: data1
 !$acc declare create(data1)
end module

! CHECK-LABEL: fir.global @_QMacc_declare_testEdata1 {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<100000xf32>

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_testEdata1_acc_ctor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_testEdata1) {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.ref<!fir.array<100000xf32>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<100000xf32>>) -> !fir.ref<!fir.array<100000xf32>> {name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100000xf32>>)
! CHECK:         acc.terminator
! CHECK:       }

! CHECK-LABEL: acc.global_dtor @_QMacc_declare_testEdata1_acc_dtor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_testEdata1) {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.ref<!fir.array<100000xf32>>
! CHECK:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<100000xf32>>) -> !fir.ref<!fir.array<100000xf32>> {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100000xf32>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.array<100000xf32>>) {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         acc.terminator
! CHECK:       }

module acc_declare_copyin_test
 integer, parameter :: n = 100000
 real, dimension(n) :: data1
 !$acc declare copyin(data1)
end module

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_copyin_testEdata1_acc_ctor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_copyin_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_copyin>} : !fir.ref<!fir.array<100000xf32>>
! CHECK:         %[[COPYIN:.*]] = acc.copyin varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<100000xf32>>) -> !fir.ref<!fir.array<100000xf32>> {name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<100000xf32>>)
! CHECK:         acc.terminator
! CHECK:       }

module acc_declare_device_resident_test
 integer, parameter :: n = 5000
 integer, dimension(n) :: data1
 !$acc declare device_resident(data1)
end module

! CHECK-LABEL: fir.global @_QMacc_declare_device_resident_testEdata1 {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>} : !fir.array<5000xi32>

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_device_resident_testEdata1_acc_ctor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_device_resident_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>} : !fir.ref<!fir.array<5000xi32>>
! CHECK:         %[[DEVICERESIDENT:.*]] = acc.declare_device_resident varPtr(%0 : !fir.ref<!fir.array<5000xi32>>) -> !fir.ref<!fir.array<5000xi32>> {name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[DEVICERESIDENT]] : !fir.ref<!fir.array<5000xi32>>)
! CHECK:         acc.terminator
! CHECK:       }

! CHECK-LABEL: acc.global_dtor @_QMacc_declare_device_resident_testEdata1_acc_dtor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_device_resident_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>} : !fir.ref<!fir.array<5000xi32>>
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<5000xi32>>)   -> !fir.ref<!fir.array<5000xi32>> {dataClause = #acc<data_clause acc_declare_device_resident>, name = "data1", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<5000xi32>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.array<5000xi32>>)   {dataClause = #acc<data_clause acc_declare_device_resident>, name = "data1", structured = false}
! CHECK:         acc.terminator
! CHECK:       }

module acc_declare_device_link_test
 integer, parameter :: n = 5000
 integer, dimension(n) :: data1
 !$acc declare link(data1)
end module

! CHECK-LABEL: fir.global @_QMacc_declare_device_link_testEdata1 {acc.declare = #acc.declare<dataClause =  acc_declare_link>} : !fir.array<5000xi32> {

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_device_link_testEdata1_acc_ctor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_device_link_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_declare_link>} : !fir.ref<!fir.array<5000xi32>>
! CHECK:         %[[LINK:.*]] = acc.declare_link varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<5000xi32>>) -> !fir.ref<!fir.array<5000xi32>> {name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[LINK]] : !fir.ref<!fir.array<5000xi32>>)
! CHECK:         acc.terminator
! CHECK:       }
