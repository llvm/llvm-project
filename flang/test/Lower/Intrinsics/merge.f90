! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmerge_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1>>{{.*}}, %[[arg1:.*]]: index{{.*}},  %[[arg2:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[arg3:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> !fir.boxchar<1> {
function merge_test(o1, o2, mask)
character :: o1, o2, merge_test
logical :: mask
merge_test = merge(o1, o2, mask)
! CHECK:  %[[a0:.*]]:2 = fir.unboxchar %[[arg2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG:  %[[a1:.*]]:2 = fir.unboxchar %[[arg3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index) 
! CHECK: %[[a2:.*]] = fir.load %[[arg4]] : !fir.ref<!fir.logical<4>>
! CHECK: %[[a3:.*]] = fir.convert %[[a2]] : (!fir.logical<4>) -> i1
! CHECK: %[[a4:.*]] = arith.select %[[a3]], %[[a0]]#0, %[[a1]]#0 : !fir.ref<!fir.char<1,?>>
! CHECK-DAG:  %{{.*}} = fir.convert %[[a4]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
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
