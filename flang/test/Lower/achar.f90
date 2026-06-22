! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Tests ACHAR lowering (converting an INTEGER to a CHARACTER (singleton, LEN=1)
! along with conversion of CHARACTER to another KIND.
subroutine achar_test1(a)
  integer, parameter :: ckind = 2
  integer, intent(in) :: a
  character(kind=ckind, len=1) :: ch

  ch = achar(a)
  call achar_test1_foo(ch)
end subroutine achar_test1

! CHECK-LABEL: func.func @_QPachar_test1(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) {
! CHECK: %[[TMP:.*]] = fir.alloca !fir.char<1>
! CHECK: %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG]] dummy_scope %[[DSCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFachar_test1Ea"}
! CHECK: %[[CH_ALLOCA:.*]] = fir.alloca !fir.char<2> {bindc_name = "ch", uniq_name = "_QFachar_test1Ech"}
! CHECK: %[[CH:.*]]:2 = hlfir.declare %[[CH_ALLOCA]] typeparams %{{.*}} {uniq_name = "_QFachar_test1Ech"}
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK: %[[A_I64:.*]] = fir.convert %[[A_VAL]] : (i32) -> i64
! CHECK: %[[A_I8:.*]] = fir.convert %[[A_I64]] : (i64) -> i8
! CHECK: %[[CHAR:.*]] = fir.insert_value %{{.*}}, %[[A_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: fir.store %[[CHAR]] to %[[TMP]] : !fir.ref<!fir.char<1>>
! CHECK: %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]
! CHECK: %[[ASSOC:.*]]:3 = hlfir.associate %[[EXPR]]
! CHECK: %[[KIND_TMP:.*]] = fir.alloca !fir.char<2,?>(%{{.*}} : index)
! CHECK: fir.char_convert %[[ASSOC]]#0 for %{{.*}} to %[[KIND_TMP]] : !fir.ref<!fir.char<1>>, index, !fir.ref<!fir.char<2,?>>
! CHECK: hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2 : !fir.ref<!fir.char<1>>, i1
! CHECK: %[[CONVERTED:.*]]:2 = hlfir.declare %[[KIND_TMP]] typeparams %{{.*}} {uniq_name = ".temp.kindconvert"}
! CHECK: %[[SET_LENGTH:.*]] = hlfir.set_length %[[CONVERTED]]#0
! CHECK: hlfir.assign %[[SET_LENGTH]] to %[[CH]]#0
! CHECK: hlfir.destroy %[[EXPR]]
! CHECK: fir.call @_QPachar_test1_foo(%{{.*}}) {{.*}}: (!fir.boxchar<2>) -> ()
