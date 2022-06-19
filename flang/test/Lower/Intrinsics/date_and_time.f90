! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdate_and_time_test(
! CHECK-SAME: %[[date:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[time:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[zone:.*]]: !fir.boxchar<1>{{.*}}, %[[values:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}) {
subroutine date_and_time_test(date, time, zone, values)
    character(*) :: date, time, zone
    integer(8) :: values(:)
    ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[timeUnbox:.*]]:2 = fir.unboxchar %[[time]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[zoneUnbox:.*]]:2 = fir.unboxchar %[[zone]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
    ! CHECK: %[[timeBuffer:.*]] = fir.convert %[[timeUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[timeLen:.*]] = fir.convert %[[timeUnbox]]#1 : (index) -> i64
    ! CHECK: %[[zoneBuffer:.*]] = fir.convert %[[zoneUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[zoneLen:.*]] = fir.convert %[[zoneUnbox]]#1 : (index) -> i64
    ! CHECK: %[[valuesCast:.*]] = fir.convert %[[values]] : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[valuesCast]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> none
    call date_and_time(date, time, zone, values)
  end subroutine
  
  ! CHECK-LABEL: func @_QPdate_and_time_test2(
  ! CHECK-SAME: %[[date:.*]]: !fir.boxchar<1>{{.*}})
  subroutine date_and_time_test2(date)
    character(*) :: date
    ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[values:.*]] = fir.absent !fir.box<none> 
    ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
    ! CHECK: %[[timeBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
    ! CHECK: %[[timeLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
    ! CHECK: %[[zoneBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
    ! CHECK: %[[zoneLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
    ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[values]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> none
    call date_and_time(date)
  end subroutine
  
  ! CHECK-LABEL: func @_QPdate_and_time_dynamic_optional(
  ! CHECK-SAME:  %[[VAL_0:[^:]*]]: !fir.boxchar<1>
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.boxchar<1>
  ! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  subroutine date_and_time_dynamic_optional(date, time, zone, values)
    ! Nothing special is required for the pointer/optional characters (the null address will
    ! directly be understood as meaning absent in the runtime). However, disassociated pointer
    ! `values` need to be transformed into an absent fir.box (nullptr descriptor address).
    character(*)  :: date
    character(:), pointer :: time
    character(*), optional :: zone
    integer, pointer :: values(:)
    call date_and_time(date, time, zone, values)
  ! CHECK:  %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK:  %[[VAL_7:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
  ! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
  ! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
  ! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_12]] : i64
  ! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    ! CHECK:  %[[VAL_15:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
  ! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_8]] : (!fir.ptr<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_22:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
  ! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
  ! CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_28:.*]] = fir.call @_FortranADateAndTime(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %{{.*}}, %{{.*}}, %[[VAL_26]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> none
  end subroutine
