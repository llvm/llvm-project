! Test GET_COMMAND_ARGUMENT with dynamically optional arguments.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:  %[[numberParam:.*]]: !fir.ref<i32> {fir.bindc_name = "number", fir.optional},
! CHECK-SAME:  %[[valueParam:.*]]: !fir.boxchar<1> {fir.bindc_name = "value", fir.optional},
! CHECK-SAME:  %[[lengthParam:.*]]: !fir.ref<i32> {fir.bindc_name = "length", fir.optional},
! CHECK-SAME:  %[[statusParam:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional},
! CHECK-SAME:  %[[errmsgParam:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg", fir.optional}) {
subroutine test(number, value, length, status, errmsg)
  integer, optional :: number, status, length
  character(*), optional :: value, errmsg
  ! Note: number cannot be absent
  call get_command_argument(number, value, length, status, errmsg)
! CHECK:  %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsgParam]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[valueParam]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[valueIsPresent:.*]] = fir.is_present {{.*}} : (!fir.boxchar<1>) -> i1
! CHECK:  %[[lengthIsPresent:.*]] = fir.is_present {{.*}} : (!fir.ref<i32>) -> i1
! CHECK:  %[[errmsgIsPresent:.*]] = fir.is_present {{.*}} : (!fir.boxchar<1>) -> i1
! CHECK:  %[[number:.*]] = fir.load {{.*}} : !fir.ref<i32>
! CHECK:  %[[valueReboxed:.*]] = fir.embox {{.*}} typeparams %[[valueUnboxed]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueOrAbsent:.*]] = arith.select %[[valueIsPresent]], %[[valueReboxed]], %[[valueAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[lengthBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:  %[[lengthAbsent:.*]] = fir.absent !fir.box<i32>
! CHECK:  %[[lengthOrAbsent:.*]] = arith.select %[[lengthIsPresent]], %[[lengthBoxed]], %[[lengthAbsent]] : !fir.box<i32>
! CHECK:  %[[errmsgReboxed:.*]] = fir.embox {{.*}} typeparams %[[errmsgUnboxed]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgOrAbsent:.*]] = arith.select %[[errmsgIsPresent]], %[[errmsgReboxed]], %[[errmsgAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK:  %[[sourceLine:.*]] = arith.constant [[# @LINE - 17]] : i32
! CHECK:  %[[value:.*]] = fir.convert %[[valueOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[length:.*]] = fir.convert %[[lengthOrAbsent]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:  %[[errmsg:.*]] = fir.convert %[[errmsgOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK:  %[[status:.*]] = fir.call @_FortranAGetCommandArgument(%[[number]], %[[value]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.store %[[status]] to {{.*}} : !fir.ref<i32>
! CHECK:  }
end subroutine
