! Test GET_ENVIRONMENT_VARIABLE with dynamically optional arguments.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:  %[[ARG_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "name", fir.optional},
! CHECK-SAME:  %[[ARG_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "value", fir.optional},
! CHECK-SAME:  %[[ARG_2:.*]]: !fir.ref<i32> {fir.bindc_name = "length", fir.optional},
! CHECK-SAME:  %[[ARG_3:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional},
! CHECK-SAME:  %[[ARG_4:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name", fir.optional},
! CHECK-SAME:  %[[ARG_5:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg", fir.optional}) {
subroutine test(name, value, length, status, trim_name, errmsg)
  integer, optional :: status, length
  character(*), optional :: name, value, errmsg
  logical, optional :: trim_name
  ! Note: name is not optional in get_environment_variable and must be present
  call get_environment_variable(name, value, length, status, trim_name, errmsg)
! CHECK:  %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG_5]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:.*]] = fir.embox %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_4:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_5:.*]] = fir.embox %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_6:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_7:.*]] = arith.select %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[ARG_2]] : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_9:.*]] = fir.embox %[[ARG_2]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:  %[[VAL_10:.*]] = fir.absent !fir.box<i32>
! CHECK:  %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !fir.box<i32>
! CHECK:  %[[VAL_12:.*]] = fir.is_present %[[VAL_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_13:.*]] = fir.embox %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_14:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_15:.*]] = arith.select %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[ARG_4]] : (!fir.ref<!fir.logical<4>>) -> i64
! CHECK:  %[[CONST_0:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_16]], %[[CONST_0]] : i64
! CHECK:  %[[VAL_18:.*]] = fir.if %[[VAL_17]] -> (i1) {
! CHECK:    %[[VAL_28:.*]] = fir.load %[[ARG_4]] : !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.logical<4>) -> i1
! CHECK:    fir.result %[[VAL_29]] : i1
! CHECK:  } else {
! CHECK:    %[[CONST_1:.*]] = arith.constant true
! CHECK:    fir.result %[[CONST_1]] : i1
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_22:.*]] = fir.convert %[[VAL_11]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_25:.*]] = fir.call @_FortranAGetEnvVariable(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_18]], %[[VAL_23]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:  %[[VAL_26:.*]] = fir.convert %[[ARG_3]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[CONST_2:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_26]], %[[CONST_2]] : i64
! CHECK:  fir.if %[[VAL_27]] {
! CHECK:    fir.store %[[VAL_25]] to %[[ARG_3]] : !fir.ref<i32>
! CHECK:  }
end subroutine
