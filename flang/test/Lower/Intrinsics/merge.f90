! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmerge_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1>>{{.*}}, %[[arg1:.*]]: index{{.*}},  %[[arg2:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[arg3:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> !fir.boxchar<1> {
function merge_test(o1, o2, mask)
character :: o1, o2, merge_test
logical :: mask
merge_test = merge(o1, o2, mask)
! CHECK:  %[[a0:.*]]:2 = fir.unboxchar %[[arg2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[a0_cast:.*]] = fir.convert %[[a0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:  %[[a1:.*]]:2 = fir.unboxchar %[[arg3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index) 
! CHECK: %[[a1_cast:.*]] = fir.convert %[[a1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK: %[[a2:.*]] = fir.load %[[arg4]] : !fir.ref<!fir.logical<4>>
! CHECK: %[[a3:.*]] = fir.convert %[[a2]] : (!fir.logical<4>) -> i1
! CHECK: %[[a4:.*]] = arith.select %[[a3]], %[[a0_cast]], %[[a1_cast]] : !fir.ref<!fir.char<1>>
! CHECK:  %{{.*}} = fir.convert %[[a4]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
end

! CHECK-LABEL: func @_QPmerge_test2(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg1:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> i32 {
function merge_test2(o1, o2, mask)
integer :: o1, o2, merge_test2
logical :: mask
merge_test2 = merge(o1, o2, mask)
! CHECK:  %[[a1:.*]] = fir.load %[[arg0]] : !fir.ref<i32>
! CHECK:  %[[a2:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
! CHECK:  %[[a3:.*]] = fir.load %[[arg2]] : !fir.ref<!fir.logical<4>>
! CHECK:  %[[a4:.*]] = fir.convert %[[a3]] : (!fir.logical<4>) -> i1
! CHECK:  %{{.*}} = arith.select %[[a4]], %[[a1]], %[[a2]] : i32
end

! CHECK-LABEL: func @_QPmerge_test3(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.ref<!fir.array<10x!fir.type<_QFmerge_test3Tt{i:i32}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.ref<!fir.type<_QFmerge_test3Tt{i:i32}>>{{.*}}, %[[arg2:[^:]+]]: !fir.ref<!fir.type<_QFmerge_test3Tt{i:i32}>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) {
subroutine merge_test3(result, o1, o2, mask)
type t
  integer :: i
end type
type(t) :: result(10), o1, o2
logical :: mask
result = merge(o1, o2, mask)
! CHECK:  %[[mask:.*]] = fir.load %[[arg3]] : !fir.ref<!fir.logical<4>>
! CHECK:  %[[mask_cast:.*]] = fir.convert %[[mask]] : (!fir.logical<4>) -> i1
! CHECK:  = arith.select %[[mask_cast]], %[[arg1]], %[[arg2]] : !fir.ref<!fir.type<_QFmerge_test3Tt{i:i32}>>
end

! CHECK-LABEL: func @_QPmerge_logical_var_and_expr(
subroutine merge_logical_var_and_expr(l1, l2)
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l1"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l2"}) {
  logical :: l1, l2
  call bar(merge(l1, .true., l2))
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.logical<4>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_4:.*]] = arith.constant true
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
! CHECK:  %[[VAL_8:.*]] = arith.select %[[VAL_6]], %[[VAL_3]], %[[VAL_7]] : !fir.logical<4>
! CHECK:  fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:  fir.call @_QPbar(%[[VAL_2]]) {{.*}}: (!fir.ref<!fir.logical<4>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPmerge_cst_and_dyn_char(
subroutine merge_cst_and_dyn_char(dyn, l)
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "dyn"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
  character(4) :: cst = "abcde"
  character(*) :: dyn
  logical :: l
  print *,  merge(cst, dyn, l)
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QFmerge_cst_and_dyn_charEcst) : !fir.ref<!fir.char<1,4>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:  %[[VAL_13:.*]] = arith.select %[[VAL_11]], %[[VAL_2]], %[[VAL_12]] : !fir.ref<!fir.char<1,4>>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
! CHECK:  fir.call @_FortranAioOutputAscii(%{{.*}}, %[[VAL_14]], %[[VAL_15]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
end subroutine
