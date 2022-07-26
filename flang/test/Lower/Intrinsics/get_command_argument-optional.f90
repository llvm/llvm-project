! Test GET_COMMAND_ARGUMENT with dynamically optional arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

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
! CHECK:  %[[number:.*]] = fir.load %[[numberParam]] : !fir.ref<i32>
! CHECK:  %[[valueIsPresent:.*]] = fir.is_present %[[valueUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[valueReboxed:.*]] = fir.embox %[[valueUnboxed]]#0 typeparams %[[valueUnboxed]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[valueOrAbsent:.*]] = arith.select %[[valueIsPresent]], %[[valueReboxed]], %[[valueAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgIsPresent:.*]] = fir.is_present %[[errmsgUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[errmsgReboxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgUnboxed]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgAbsent:.*]] = fir.absent !fir.box<!fir.char<1,?>>
! CHECK:  %[[errmsgOrAbsent:.*]] = arith.select %[[errmsgIsPresent]], %[[errmsgReboxed]], %[[errmsgAbsent]] : !fir.box<!fir.char<1,?>>
! CHECK:  %[[value:.*]] = fir.convert %[[valueOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[errmsg:.*]] = fir.convert %[[errmsgOrAbsent]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:  %[[status:.*]] = fir.call @_FortranAArgumentValue(%[[number]], %[[value]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK:  %[[statusI64:.*]] = fir.convert %[[statusParam]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[zero:.*]] = arith.constant 0 : i64
! CHECK:  %[[statusIsNonNull:.*]] = arith.cmpi ne, %[[statusI64]], %[[zero]] : i64
! CHECK:  fir.if %[[statusIsNonNull]] {
! CHECK:    fir.store %[[status]] to %[[statusParam]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  %[[lengthI64:.*]] = fir.convert %[[lengthParam]] : (!fir.ref<i32>) -> i64
! CHECK:  %[[zero2:.*]] = arith.constant 0 : i64
! CHECK:  %[[lengthIsNonNull:.*]] = arith.cmpi ne, %[[lengthI64]], %[[zero2]] : i64
! CHECK:  fir.if %[[lengthIsNonNull]] {
! CHECK:    %[[length:.*]] = fir.call @_FortranAArgumentLength(%[[number]]) : (i32) -> i64
! CHECK:    %[[lengthTrunc:.*]] = fir.convert %[[length]] : (i64) -> i32
! CHECK:    fir.store %[[lengthTrunc]] to %[[lengthParam]] : !fir.ref<i32>
! CHECK:  }
end subroutine
