! Test GET_COMMAND_ARGUMENT with dynamically optional arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "number", fir.optional},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "value", fir.optional},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "length", fir.optional},
! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional},
! CHECK-SAME:  %[[VAL_4:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg", fir.optional}) {
subroutine test(number, value, length, status, errmsg) 
  integer, optional :: number, status, length
  character(*), optional :: value, errmsg
  ! Note: number cannot be absent
  call get_command_argument(number, value, length, status, errmsg) 
! CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_4]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_9:.*]] = fir.embox %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_10:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_12:.*]] = fir.is_present %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_13:.*]] = fir.embox %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_14:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_15:.*]] = arith.select %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_18:.*]] = fir.call @_FortranAArgumentValue(%[[VAL_7]], %[[VAL_16]], %[[VAL_17]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_19]], %[[VAL_20]] : i64
! CHECK:  fir.if %[[VAL_21]] {
! CHECK:    fir.store %[[VAL_18]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  %[[VAL_22:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[VAL_23:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_24:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_23]] : i64
! CHECK:  fir.if %[[VAL_24]] {
! CHECK:    %[[VAL_25:.*]] = fir.call @_FortranAArgumentLength(%[[VAL_7]]) : (i32) -> i64
! CHECK:    %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> i32
! CHECK:    fir.store %[[VAL_26]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:  }
end subroutine
