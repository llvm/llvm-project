! Test GET_ENVIRONMENT_VARIABLE with dynamically optional arguments.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s


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
! CHECK:  %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[ARG_5]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[errmsgDeclare:.*]]:2 = hlfir.declare %[[errmsgUnbox]]#0 typeparams %[[errmsgUnbox]]#1
! CHECK:  %[[lengthDeclare:.*]]:2 = hlfir.declare %[[ARG_2]]
! CHECK:  %[[nameUnbox:.*]]:2 = fir.unboxchar %[[ARG_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[nameDeclare:.*]]:2 = hlfir.declare %[[nameUnbox]]#0 typeparams %[[nameUnbox]]#1
! CHECK:  %[[statusDeclare:.*]]:2 = hlfir.declare %[[ARG_3]]
! CHECK:  %[[trimNameDeclare:.*]]:2 = hlfir.declare %[[ARG_4]]
! CHECK:  %[[valueUnbox:.*]]:2 = fir.unboxchar %[[ARG_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[valueDeclare:.*]]:2 = hlfir.declare %[[valueUnbox]]#0 typeparams %[[valueUnbox]]#1
! CHECK:  %[[valueIsPresent:.*]] = fir.is_present %[[valueDeclare]]#0 : (!fir.boxchar<1>) -> i1
! CHECK:  %[[lengthIsPresent:.*]] = fir.is_present %[[lengthDeclare]]#0 : (!fir.ref<i32>) -> i1
! CHECK:  %[[errmsgIsPresent:.*]] = fir.is_present %[[errmsgDeclare]]#0 : (!fir.boxchar<1>) -> i1
! CHECK:  %[[nameBox:.*]] = fir.embox %[[nameDeclare]]#1 typeparams %[[nameUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueReboxed:.*]] = fir.embox %[[valueDeclare]]#1 typeparams %[[valueUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueOrAbsent:.*]] = arith.select %[[valueIsPresent]], %[[valueReboxed]], %[[valueAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[lengthBox:.*]] = fir.embox %[[lengthDeclare]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:  %[[lengthAbsent:.*]] = fir.absent !fir.box<i32>
! CHECK:  %[[lengthOrAbsent:.*]] = arith.select %[[lengthIsPresent]], %[[lengthBox]], %[[lengthAbsent]] : !fir.box<i32>
! CHECK:  %[[errmsgReboxed:.*]] = fir.embox %[[errmsgDeclare]]#1 typeparams %[[errmsgUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgOrAbsent:.*]] = arith.select %[[errmsgIsPresent]], %[[errmsgReboxed]], %[[errmsgAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[trimName:.*]] = fir.if %{{.*}} -> (i1) {
! CHECK:    %[[trimLoad:.*]] = fir.load %[[trimNameDeclare]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:    %[[trimCast:.*]] = fir.convert %[[trimLoad]] : (!fir.logical<4>) -> i1
! CHECK:    fir.result %[[trimCast]] : i1
! CHECK:  } else {
! CHECK:    %[[trueVal:.*]] = arith.constant true
! CHECK:    fir.result %[[trueVal]] : i1
! CHECK:  }
! CHECK:  %[[sourceLine:.*]] = arith.constant 17 : i32
! CHECK:  %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[value:.*]] = fir.convert %[[valueOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[length:.*]] = fir.convert %[[lengthOrAbsent]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:  %[[errmsg:.*]] = fir.convert %[[errmsgOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %{{.*}}, %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.store %[[status]] to %[[statusDeclare]]#0 : !fir.ref<i32>
! CHECK:  }
end subroutine
