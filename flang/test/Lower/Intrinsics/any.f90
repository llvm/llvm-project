! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: any_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}) -> !fir.logical<4>
logical function any_test(mask)
logical :: mask(:)
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[a1:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[a2:.*]] = fir.convert %[[c1]] : (index) -> i32
any_test = any(mask)
! CHECK:  %[[a3:.*]] = fir.call @_FortranAAny(%[[a1]], %{{.*}},  %{{.*}}, %[[a2]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
end function any_test

! CHECK-LABEL: any_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine any_test2(mask, d, rslt)
logical :: mask(:,:)
integer :: d
logical :: rslt(:)
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %[[arg1:.*]] : !fir.ref<i32>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0:.*]]: (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
rslt = any(mask, d)
! CHECK:  %[[r1:.*]] = fir.call @_FortranAAnyDim(%[[a6:.*]], %[[a7:.*]], %[[a1:.*]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a10:.*]] = fir.load %[[a0:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK-DAG:  %[[a12:.*]] = fir.box_addr %[[a10:.*]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK-DAG  fir.freemem %[[a12:.*]]
end subroutine
