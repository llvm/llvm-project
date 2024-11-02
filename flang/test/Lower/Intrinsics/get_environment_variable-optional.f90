! Test GET_ENVIRONMENT_VARIABLE with dynamically optional arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "name", fir.optional},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "value", fir.optional},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "length", fir.optional},
! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional},
! CHECK-SAME:  %[[VAL_4:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name", fir.optional},
! CHECK-SAME:  %[[VAL_5:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg", fir.optional}) {
subroutine test(name, value, length, status, trim_name, errmsg) 
  integer, optional :: status, length
  character(*), optional :: name, value, errmsg
  logical, optional :: trim_name
  ! Note: name is not optional in et_environment_variable and must be present
  call get_environment_variable(name, value, length, status, trim_name, errmsg) 
! CHECK:  %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_5]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_9:.*]] = fir.embox %[[VAL_7]]#0 typeparams %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_10:.*]] = fir.is_present %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_11:.*]] = fir.embox %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_12:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_13:.*]] = arith.select %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_14:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_15:.*]] = fir.embox %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_16:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.logical<4>>) -> i64
! CHECK:  %[[VAL_19:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_19]] : i64
! CHECK:  %[[VAL_21:.*]] = fir.if %[[VAL_20]] -> (i1) {
! CHECK:    %[[VAL_22:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.logical<4>) -> i1
! CHECK:    fir.result %[[VAL_23]] : i1
! CHECK:  } else {
! CHECK:    %[[VAL_24:.*]] = arith.constant true
! CHECK:    fir.result %[[VAL_24]] : i1
! CHECK:  }
! CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_28:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_29:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[VAL_31:.*]] = fir.call @_FortranAEnvVariableValue(%[[VAL_27]], %[[VAL_28]], %[[VAL_32:.*]], %[[VAL_29]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:  %[[VAL_33:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[VAL_34:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_35:.*]] = arith.cmpi ne, %[[VAL_33]], %[[VAL_34]] : i64
! CHECK:  fir.if %[[VAL_35]] {
! CHECK:    fir.store %[[VAL_31]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  %[[VAL_36:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[VAL_37:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_36]], %[[VAL_37]] : i64
! CHECK:  fir.if %[[VAL_38]] {
! CHECK:    %[[VAL_41:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:    %[[VAL_43:.*]] = fir.call @_FortranAEnvVariableLength(%[[VAL_41]], %[[VAL_32]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i1, !fir.ref<i8>, i32) -> i64
! CHECK:    %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i64) -> i32
! CHECK:    fir.store %[[VAL_44]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:  }
end subroutine
