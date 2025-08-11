!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device %s -o - | FileCheck %s

module test_0
    implicit none

!CHECK-DAG: fir.global @_QMtest_0Edata_int {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : i32
INTEGER :: data_int = 10
!$omp declare target link(data_int)

!CHECK-DAG: fir.global @_QMtest_0Earray_1d({{.*}}) {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : !fir.array<3xi32>
INTEGER :: array_1d(3) = (/1,2,3/)
!$omp declare target link(array_1d)

!CHECK-DAG: fir.global @_QMtest_0Earray_2d({{.*}}) {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : !fir.array<2x2xi32>
INTEGER :: array_2d(2,2) = reshape((/1,2,3,4/), (/2,2/))
!$omp declare target link(array_2d)

!CHECK-DAG: fir.global @_QMtest_0Ept1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : !fir.box<!fir.ptr<i32>>
INTEGER, POINTER :: pt1
!$omp declare target link(pt1)

!CHECK-DAG: fir.global @_QMtest_0Ept2_tar {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} target : i32
INTEGER, TARGET :: pt2_tar = 5
!$omp declare target link(pt2_tar)

!CHECK-DAG: fir.global @_QMtest_0Ept2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : !fir.box<!fir.ptr<i32>>
INTEGER, POINTER :: pt2 => pt2_tar
!$omp declare target link(pt2)

!CHECK-DAG: fir.global @_QMtest_0Edata_int_to {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : i32
INTEGER :: data_int_to = 5
!$omp declare target to(data_int_to)

!CHECK-DAG: fir.global @_QMtest_0Edata_int_enter {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} : i32
INTEGER :: data_int_enter = 5
!$omp declare target enter(data_int_enter)

!CHECK-DAG: fir.global @_QMtest_0Edata_int_clauseless {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : i32
INTEGER :: data_int_clauseless = 1
!$omp declare target(data_int_clauseless)

!CHECK-DAG: fir.global @_QMtest_0Edata_extended_to_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : f32
!CHECK-DAG: fir.global @_QMtest_0Edata_extended_to_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : f32
REAL :: data_extended_to_1 = 2
REAL :: data_extended_to_2 = 3
!$omp declare target to(data_extended_to_1, data_extended_to_2)

!CHECK-DAG: fir.global @_QMtest_0Edata_extended_enter_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} : f32
!CHECK-DAG: fir.global @_QMtest_0Edata_extended_enter_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} : f32
REAL :: data_extended_enter_1 = 2
REAL :: data_extended_enter_2 = 3
!$omp declare target enter(data_extended_enter_1, data_extended_enter_2)

!CHECK-DAG: fir.global @_QMtest_0Edata_extended_link_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : f32
!CHECK-DAG: fir.global @_QMtest_0Edata_extended_link_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : f32
REAL :: data_extended_link_1 = 2
REAL :: data_extended_link_2 = 3
!$omp declare target link(data_extended_link_1, data_extended_link_2)

!CHECK-DAG: fir.global @_QMtest_0Eautomap_data {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = true>} target : !fir.box<!fir.heap<i32>>
INTEGER, ALLOCATABLE, TARGET :: automap_data
!$omp declare target enter(automap : automap_data)

contains
end module test_0

PROGRAM commons
    !CHECK-DAG: fir.global @numbers_ {alignment = 4 : i64, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : tuple<f32, f32> {
    REAL :: one = 1
    REAL :: two = 2
    COMMON /numbers/ one, two
    !$omp declare target(/numbers/)

    !CHECK-DAG: fir.global @numbers_link_ {alignment = 4 : i64, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : tuple<f32, f32> {
    REAL :: one_link = 1
    REAL :: two_link = 2
    COMMON /numbers_link/ one_link, two_link
    !$omp declare target link(/numbers_link/)

    !CHECK-DAG: fir.global @numbers_to_ {alignment = 4 : i64, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : tuple<f32, f32> {
    REAL :: one_to = 1
    REAL :: two_to = 2
    COMMON /numbers_to/ one_to, two_to
    !$omp declare target to(/numbers_to/)

    !CHECK-DAG: fir.global @numbers_enter_ {alignment = 4 : i64, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} : tuple<f32, f32> {
    REAL :: one_enter = 1
    REAL :: two_enter = 2
    COMMON /numbers_enter/ one_enter, two_enter
    !$omp declare target enter(/numbers_enter/)
END
