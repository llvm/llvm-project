! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPparity_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}) -> !fir.logical<4>
logical function parity_test(mask)
  logical :: mask(:)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[a1:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[a2:.*]] = fir.convert %[[c1]] : (index) -> i32
  parity_test = parity(mask)
! CHECK:  %[[a3:.*]] = fir.call @_FortranAParity(%[[a1]], %{{.*}}, %{{.*}}, %[[a2]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
end function parity_test

! CHECK-LABEL: func.func @_QPparity_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine parity_test2(mask, d, rslt)
  logical :: mask(:,:)
  integer :: d
  logical :: rslt(:)
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK-DAG:  %[[D:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG:  %[[a1:.*]] = fir.load %[[D]]#0 : !fir.ref<i32>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  rslt = parity(mask, d)
! CHECK:  fir.call @_FortranAParityDim(%[[a6]], %[[a7]], %[[a1]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
end subroutine parity_test2
