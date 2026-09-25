! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmerge_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.char<1>>{{.*}}, %[[ARG1:.*]]: index{{.*}},  %[[ARG2:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ARG3:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ARG4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> !fir.boxchar<1> {
function merge_test(o1, o2, mask)
character :: o1, o2, merge_test
logical :: mask
merge_test = merge(o1, o2, mask)
! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmerge_testEmask"}
! CHECK-DAG:  %[[RES:.*]]:2 = hlfir.declare %[[ARG0]] typeparams %{{.*}} {uniq_name = "_QFmerge_testEmerge_test"}
! CHECK:  %[[O1_UNBOX:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[O1_CAST:.*]] = fir.convert %[[O1_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:  %[[O1:.*]]:2 = hlfir.declare %[[O1_CAST]] typeparams %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmerge_testEo1"}
! CHECK:  %[[O2_UNBOX:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[O2_CAST:.*]] = fir.convert %[[O2_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:  %[[O2:.*]]:2 = hlfir.declare %[[O2_CAST]] typeparams %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmerge_testEo2"}
! CHECK:  %[[MASK_VAL:.*]] = fir.load %[[MASK]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[MASK_BOOL:.*]] = fir.convert %[[MASK_VAL]] : (!fir.logical<4>) -> i1
! CHECK:  %[[SEL:.*]] = arith.select %[[MASK_BOOL]], %[[O1]]#0, %[[O2]]#0 : !fir.ref<!fir.char<1>>
! CHECK:  %[[TMP:.*]]:2 = hlfir.declare %[[SEL]] typeparams %{{.*}} {uniq_name = ".tmp.intrinsic_result"}
! CHECK:  %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 : (!fir.ref<!fir.char<1>>) -> !hlfir.expr<!fir.char<1>>
! CHECK:  hlfir.assign %[[EXPR]] to %[[RES]]#0 : !hlfir.expr<!fir.char<1>>, !fir.ref<!fir.char<1>>
! CHECK:  hlfir.destroy %[[EXPR]] : !hlfir.expr<!fir.char<1>>
end

! CHECK-LABEL: func @_QPmerge_test2(
! CHECK-SAME: %[[ARG0:[^:]+]]: !fir.ref<i32>{{.*}}, %[[ARG1:[^:]+]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> i32 {
function merge_test2(o1, o2, mask)
integer :: o1, o2, merge_test2
logical :: mask
merge_test2 = merge(o1, o2, mask)
! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmerge_test2Emask"}
! CHECK-DAG:  %[[O1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmerge_test2Eo1"}
! CHECK-DAG:  %[[O2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmerge_test2Eo2"}
! CHECK:  %[[O1_VAL:.*]] = fir.load %[[O1]]#0 : !fir.ref<i32>
! CHECK:  %[[O2_VAL:.*]] = fir.load %[[O2]]#0 : !fir.ref<i32>
! CHECK:  %[[MASK_VAL:.*]] = fir.load %[[MASK]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[MASK_BOOL:.*]] = fir.convert %[[MASK_VAL]] : (!fir.logical<4>) -> i1
! CHECK:  %{{.*}} = arith.select %[[MASK_BOOL]], %[[O1_VAL]], %[[O2_VAL]] : i32
end

! CHECK-LABEL: func @_QPmerge_test3(
subroutine merge_test3(result, o1, o2, mask)
type t
  integer :: i
end type
type(t) :: result(10), o1, o2
logical :: mask
result = merge(o1, o2, mask)
! CHECK:  %[[MASK:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 4 {uniq_name = "_QFmerge_test3Emask"}
! CHECK:  %[[O1:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmerge_test3Eo1"}
! CHECK:  %[[O2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmerge_test3Eo2"}
! CHECK:  %[[MASK_VAL:.*]] = fir.load %[[MASK]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[MASK_BOOL:.*]] = fir.convert %[[MASK_VAL]] : (!fir.logical<4>) -> i1
! CHECK:  %[[SEL:.*]] = arith.select %[[MASK_BOOL]], %[[O1]]#0, %[[O2]]#0 : !fir.ref<!fir.type<_QFmerge_test3Tt{i:i32}>>
! CHECK:  %[[TMP:.*]]:2 = hlfir.declare %[[SEL]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:  %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 : (!fir.ref<!fir.type<_QFmerge_test3Tt{i:i32}>>) -> !hlfir.expr<!fir.type<_QFmerge_test3Tt{i:i32}>>
! CHECK:  hlfir.assign %[[EXPR]] to %{{.*}}#0 : !hlfir.expr<!fir.type<_QFmerge_test3Tt{i:i32}>>, !fir.ref<!fir.array<10x!fir.type<_QFmerge_test3Tt{i:i32}>>>
! CHECK:  hlfir.destroy %[[EXPR]] : !hlfir.expr<!fir.type<_QFmerge_test3Tt{i:i32}>>
end

! CHECK-LABEL: func @_QPmerge_logical_var_and_expr(
subroutine merge_logical_var_and_expr(l1, l2)
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l1"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l2"}) {
  logical :: l1, l2
  call bar(merge(l1, .true., l2))
! CHECK-DAG:  %[[L1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmerge_logical_var_and_exprEl1"}
! CHECK-DAG:  %[[L2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmerge_logical_var_and_exprEl2"}
! CHECK:  %[[TRUE:.*]] = arith.constant true
! CHECK:  %[[L1_VAL:.*]] = fir.load %[[L1]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[L2_VAL:.*]] = fir.load %[[L2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[L2_BOOL:.*]] = fir.convert %[[L2_VAL]] : (!fir.logical<4>) -> i1
! CHECK:  %[[TRUE_LOG:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
! CHECK:  %[[SEL:.*]] = arith.select %[[L2_BOOL]], %[[L1_VAL]], %[[TRUE_LOG]] : !fir.logical<4>
! CHECK:  %[[ASSOC:.*]]:3 = hlfir.associate %[[SEL]] {adapt.valuebyref}
! CHECK:  fir.call @_QPbar(%[[ASSOC]]#0) {{.*}}: (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:  hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2
end subroutine

! CHECK-LABEL: func @_QPmerge_cst_and_dyn_char(
subroutine merge_cst_and_dyn_char(dyn, l)
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "dyn"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
  character(4) :: cst = "abcde"
  character(*) :: dyn
  logical :: l
  print *,  merge(cst, dyn, l)
! CHECK:  %[[CST:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = "_QFmerge_cst_and_dyn_charEcst"}
! CHECK:  %[[DYN_UNBOX:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[DYN:.*]]:2 = hlfir.declare %[[DYN_UNBOX]]#0 typeparams %[[DYN_UNBOX]]#1 dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmerge_cst_and_dyn_charEdyn"}
! CHECK:  %[[L:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmerge_cst_and_dyn_charEl"}
! CHECK:  %[[L_VAL:.*]] = fir.load %[[L]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[L_BOOL:.*]] = fir.convert %[[L_VAL]] : (!fir.logical<4>) -> i1
! CHECK:  %[[DYN_CAST:.*]] = fir.convert %[[DYN]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:  %[[SEL:.*]] = arith.select %[[L_BOOL]], %[[CST]]#0, %[[DYN_CAST]] : !fir.ref<!fir.char<1,4>>
! CHECK:  %[[TMP:.*]]:2 = hlfir.declare %[[SEL]] typeparams %{{.*}} {uniq_name = ".tmp.intrinsic_result"}
! CHECK:  %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 : (!fir.ref<!fir.char<1,4>>) -> !hlfir.expr<!fir.char<1,4>>
! CHECK:  %[[ASSOC:.*]]:3 = hlfir.associate %[[EXPR]] typeparams %{{.*}} {adapt.valuebyref}
! CHECK:  %[[ASSOC_CAST:.*]] = fir.convert %[[ASSOC]]#0 : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:  fir.call @_FortranAioOutputAscii(%{{.*}}, %[[ASSOC_CAST]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:  hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2
! CHECK:  hlfir.destroy %[[EXPR]] : !hlfir.expr<!fir.char<1,4>>
end subroutine
