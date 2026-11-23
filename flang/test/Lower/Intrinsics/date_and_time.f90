! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdate_and_time_test(
! CHECK-SAME: %[[date:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[time:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[zone:.*]]: !fir.boxchar<1>{{.*}}, %[[values:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}) {
subroutine date_and_time_test(date, time, zone, values)
    character(*) :: date, time, zone
    integer(8) :: values(:)
    ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
    ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[dateDecl:.*]]:2 = hlfir.declare %[[dateUnbox]]#0 typeparams %[[dateUnbox]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[timeUnbox:.*]]:2 = fir.unboxchar %[[time]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[timeDecl:.*]]:2 = hlfir.declare %[[timeUnbox]]#0 typeparams %[[timeUnbox]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[valuesDecl:.*]]:2 = hlfir.declare %[[values]] dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[zoneUnbox:.*]]:2 = fir.unboxchar %[[zone]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[zoneDecl:.*]]:2 = hlfir.declare %[[zoneUnbox]]#0 typeparams %[[zoneUnbox]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateDecl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
    ! CHECK: %[[timeBuffer:.*]] = fir.convert %[[timeDecl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[timeLen:.*]] = fir.convert %[[timeUnbox]]#1 : (index) -> i64
    ! CHECK: %[[zoneBuffer:.*]] = fir.convert %[[zoneDecl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[zoneLen:.*]] = fir.convert %[[zoneUnbox]]#1 : (index) -> i64
    ! CHECK: %[[valuesCast:.*]] = fir.convert %[[valuesDecl]]#1 : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[valuesCast]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> ()
    call date_and_time(date, time, zone, values)
  end subroutine

  ! CHECK-LABEL: func @_QPdate_and_time_test2(
  ! CHECK-SAME: %[[date:.*]]: !fir.boxchar<1>{{.*}})
  subroutine date_and_time_test2(date)
    character(*) :: date
    ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
    ! CHECK: %[[dateUnbox:.*]]:2 = fir.unboxchar %[[date]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[dateDecl:.*]]:2 = hlfir.declare %[[dateUnbox]]#0 typeparams %[[dateUnbox]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[values:.*]] = fir.absent !fir.box<none>
    ! CHECK: %[[dateBuffer:.*]] = fir.convert %[[dateDecl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[dateLen:.*]] = fir.convert %[[dateUnbox]]#1 : (index) -> i64
    ! CHECK: %[[timeBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
    ! CHECK: %[[timeLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
    ! CHECK: %[[zoneBuffer:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.ref<i8>
    ! CHECK: %[[zoneLen:.*]] = fir.convert %c0{{.*}} : (index) -> i64
    ! CHECK: fir.call @_FortranADateAndTime(%[[dateBuffer]], %[[dateLen]], %[[timeBuffer]], %[[timeLen]], %[[zoneBuffer]], %[[zoneLen]], %{{.*}}, %{{.*}}, %[[values]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> ()
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
    ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
    ! CHECK: %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[VAL_date:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[VAL_time:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[VAL_values:.*]]:2 = hlfir.declare %[[VAL_3]] dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[VAL_zone:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[DS]] {{.*}}
    ! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_values]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    ! CHECK: %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
    ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
    ! CHECK: %[[VAL_10:.*]] = arith.constant 0 : i64
    ! CHECK: %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_10]] : i64
    ! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_time]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
    ! CHECK: %[[VAL_13:.*]] = fir.box_elesize %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
    ! CHECK: %[[VAL_14:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
    ! CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_values]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    ! CHECK: %[[VAL_16:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
    ! CHECK: %[[VAL_17:.*]] = arith.select %[[VAL_11]], %[[VAL_15]], %[[VAL_16]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
    ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_date]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
    ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_14]] : (!fir.ptr<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
    ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_zone]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK: %[[VAL_24:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
    ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranADateAndTime(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %{{.*}}, %{{.*}}, %[[VAL_26]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.ref<i8>, i32, !fir.box<none>) -> ()
    call date_and_time(date, time, zone, values)
  end subroutine
