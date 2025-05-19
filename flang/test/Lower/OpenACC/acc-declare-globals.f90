! This test checks lowering of OpenACC declare directive in module specification
! part.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_declare_common_test
! CHECK-LABEL: fir.global @numbers_ {acc.declare = #acc.declare<dataClause = acc_declare_device_resident>, alignment = 4 : i64} : tuple<f32, f32> {
! CHECK: acc.global_ctor @numbers__acc_ctor
! CHECK: acc.global_dtor @numbers__acc_dtor
 REAL :: one = 1
 REAL :: two = 2
 COMMON /numbers/ one, two
 !$acc declare device_resident(/numbers/)

! CHECK-LABEL: fir.global @numbers_create_ {acc.declare = #acc.declare<dataClause = acc_create>, alignment = 4 : i64} : tuple<f32, f32> {
! CHECK: acc.global_ctor @numbers_create__acc_ctor
! CHECK: acc.global_dtor @numbers_create__acc_dtor
 REAL :: one_create = 1
 REAL :: two_create = 2
 COMMON /numbers_create/ one_create, two_create
 !$acc declare create(/numbers_create/)

! CHECK-LABEL: fir.global @numbers_in_ {acc.declare = #acc.declare<dataClause = acc_copyin>, alignment = 4 : i64} : tuple<f32, f32> {
! CHECK: acc.global_ctor @numbers_in__acc_ctor
! CHECK: acc.global_dtor @numbers_in__acc_dtor
 REAL :: one_in = 1
 REAL :: two_in = 2
 COMMON /numbers_in/ one_in, two_in
 !$acc declare copyin(/numbers_in/)

! CHECK-LABEL: fir.global @numbers_link_ {acc.declare = #acc.declare<dataClause = acc_declare_link>, alignment = 4 : i64} : tuple<f32, f32> {
! CHECK: acc.global_ctor @numbers_link__acc_ctor
 REAL :: one_link = 1
 REAL :: two_link = 2
 COMMON /numbers_link/ one_link, two_link
 !$acc declare link(/numbers_link/)
end module

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

! CHECK-LABEL: acc.global_dtor @_QMacc_declare_copyin_testEdata1_acc_dtor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_copyin_testEdata1) {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.ref<!fir.array<100000xf32>>
! CHECK:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.array<100000xf32>>) -> !fir.ref<!fir.array<100000xf32>> {dataClause = #acc<data_clause acc_copyin>, name = "data1", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100000xf32>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.array<100000xf32>>) {dataClause = #acc<data_clause acc_copyin>, name = "data1", structured = false}
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
