! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPichar_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK-DAG: %[[UNBOX:.*]]:2 = fir.unboxchar %[[ARG0]]
  ! CHECK-DAG: %[[CONV:.*]] = fir.convert %[[UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
  ! CHECK-DAG: %[[C_DECL:.*]]:2 = hlfir.declare %[[CONV]] typeparams {{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFichar_testEc"}
  ! CHECK-DAG: %[[J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFichar_testEj"}
  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[J_ALLOC]] {uniq_name = "_QFichar_testEj"}
  ! CHECK-DAG: %[[STR_ALLOC:.*]] = fir.alloca !fir.array<10x!fir.char<1>> {bindc_name = "str", uniq_name = "_QFichar_testEstr"}
  ! CHECK-DAG: %[[STR_DECL:.*]]:2 = hlfir.declare %[[STR_ALLOC]]({{.*}}) typeparams {{.*}} {uniq_name = "_QFichar_testEstr"}

  ! CHECK: %[[C_VAL:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.extract_value %[[C_VAL]], [0 : index] : (!fir.char<1>) -> i8
  ! CHECK: %[[ARG:.*]] = arith.extui %[[CHAR]] : i8 to i32
  ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[ARG]])
  ! CHECK: fir.call @_FortranAioEndIoStatement
  print *, ichar(c)

  ! CHECK: %[[J_VAL:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[J_IDX:.*]] = fir.convert %[[J_VAL]] : (i32) -> i64
  ! CHECK: %[[STR_EL:.*]] = hlfir.designate %[[STR_DECL]]#0 (%[[J_IDX]])  typeparams {{.*}} : (!fir.ref<!fir.array<10x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[STR_VAL:.*]] = fir.load %[[STR_EL]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR2:.*]] = fir.extract_value %[[STR_VAL]], [0 : index] : (!fir.char<1>) -> i8
  ! CHECK: %[[ARG2:.*]] = arith.extui %[[CHAR2]] : i8 to i32
  ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[ARG2]])
  ! CHECK: fir.call @_FortranAioEndIoStatement
  print *, ichar(str(J))

  ! "Magic" 88 below is the ASCII code for `X` and the value returned by IACHAR (’X’)
  ! CHECK: %[[c88:.*]] = arith.constant 88 : i32
  ! CHECK: fir.call @_FortranAioOutputInteger32({{.*}}, %[[c88]])
  ! CHECK: fir.call @_FortranAioEndIoStatement
  print *, iachar('X')
end subroutine

! Check that 'arith.extui' op is not generated if type are matching.
! CHECK-LABEL: func.func @_QPno_extui(
subroutine no_extui(ch)
  integer, parameter :: kind = selected_char_kind('ISO_10646')
  character(*, kind), intent(in) :: ch(:)
  integer :: i, j
  ! CHECK-NOT: arith.extui
  ! CHECK: %[[CHAR4:.*]] = fir.extract_value {{.*}}, [0 : index] : (!fir.char<4>) -> i32
  ! CHECK: hlfir.assign %[[CHAR4]] to {{.*}} : i32, !fir.ref<i32>
  j = ichar(ch(i)(i:i))
end subroutine
