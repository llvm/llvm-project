! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:  func.func @_QPtrim_test(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:          %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-NEXT:     %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:     %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFtrim_testEc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 8 : index
! CHECK-NEXT:     %[[VAL_5:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "tc", uniq_name = "_QFtrim_testEtc"}
! CHECK-NEXT:     %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_4]] {uniq_name = "_QFtrim_testEtc"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
! CHECK-NEXT:     %[[VAL_7:.*]] = hlfir.char_trim %[[VAL_3]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK-NEXT:     hlfir.assign %[[VAL_7]] to %[[VAL_6]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.char<1,8>>
! CHECK-NEXT:     hlfir.destroy %[[VAL_7]] : !hlfir.expr<!fir.char<1,?>>
! CHECK-NEXT:     return
! CHECK-NEXT:   }
subroutine trim_test(c)
  character(*) :: c
  character(8) :: tc

  tc = trim(c)
end subroutine

! Test trim with fixed length character.
! The length of the returned character type must be unknown.
! CHECK-LABEL:  func.func @_QPtrim_test2(
! CHECK:          hlfir.char_trim %{{.*}}#0 : (!fir.ref<!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,?>>
subroutine trim_test2(c)
  character(8) :: c
  character(8) :: tc

  tc = trim(c)
end subroutine
