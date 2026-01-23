! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: all_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}) -> !fir.logical<4>
logical function all_test(mask)
logical :: mask(:)
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[a1:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[a2:.*]] = fir.convert %[[c1]] : (index) -> i32
all_test = all(mask)
! CHECK:  %[[a3:.*]] = fir.call @_FortranAAll(%[[a1]], %{{.*}}, %{{.*}}, %[[a2]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
end function all_test

! CHECK-LABEL: all_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine all_test2(mask, d, rslt)
logical :: mask(:,:)
integer :: d
logical :: rslt(:)
! CHECK:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[a1:.*]] = fir.load %[[arg1:.*]] : !fir.ref<i32>
! CHECK:  %[[a6:.*]] = fir.convert %[[a0:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[a7:.*]] = fir.convert %[[arg0:.*]]: (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
rslt = all(mask, d)
! CHECK:  fir.call @_FortranAAllDim(%[[a6:.*]], %[[a7:.*]], %[[a1:.*]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
! CHECK:  %[[a10:.*]] = fir.load %[[a0:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[a12:.*]] = fir.box_addr %[[a10:.*]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:  fir.freemem %[[a12:.*]]
end subroutine
