! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Tests ACHAR lowering (converting an INTEGER to a CHARACTER (singleton, LEN=1)
! along with conversion of CHARACTER to another KIND.
subroutine achar_test1(a)
  integer, parameter :: ckind = 2
  integer, intent(in) :: a
  character(kind=ckind, len=1) :: ch

  ch = achar(a)
  call achar_test1_foo(ch)
end subroutine achar_test1

! CHECK-LABEL: func @_QPachar_test1(
! CHECK-SAME: %[[arg:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) {
! CHECK: %[[VAL_TMP_ALLOCA:.*]] = fir.alloca !fir.char<2> {bindc_name = ".tmp"}
! CHECK: %[[VAL_1_ALLOCA:.*]] = fir.alloca !fir.char<1>
! CHECK: %[[VAL_DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_DECLARE:.*]] = fir.declare %[[arg]] dummy_scope %[[VAL_DUMMY_SCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFachar_test1Ea"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK: %[[VAL_CH_ALLOCA:.*]] = fir.alloca !fir.char<2> {bindc_name = "ch", uniq_name = "_QFachar_test1Ech"}
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_DECLARE]] : !fir.ref<i32>
! CHECK: %[[VAL_5:.*]] = fir.undefined !fir.char<1>
! CHECK: %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %{{.*}}, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: fir.store %[[VAL_6]] to %[[VAL_1_ALLOCA]] : !fir.ref<!fir.char<1>>
! CHECK: %[[VAL_7:.*]] = fir.alloca !fir.char<2,?>(%{{.*}} : index)
! CHECK: fir.char_convert %[[VAL_1_ALLOCA]] for %{{.*}} to %[[VAL_7]] : !fir.ref<!fir.char<1>>, index, !fir.ref<!fir.char<2,?>>
! CHECK: fir.call @_QPachar_test1_foo(%{{.*}}) {{.*}}: (!fir.boxchar<2>) -> ()
