! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: count_test1
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
subroutine count_test1(rslt, mask)
    integer :: rslt
    logical :: mask(:)
  ! CHECK-DAG:  %[[c1:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[a2:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK:  %[[a4:.*]] = fir.convert %[[c1]] : (index) -> i32
    rslt = count(mask)
  ! CHECK:  %[[a5:.*]] = fir.call @_FortranACount(%[[a2]], %{{.*}}, %{{.*}}, %[[a4]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
  end subroutine
  
  ! CHECK-LABEL: test_count2
  ! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}})
  subroutine test_count2(rslt, mask)
    integer :: rslt(:)
    logical :: mask(:,:)
  ! CHECK-DAG:  %[[c1_i32:.*]] = arith.constant 1 : i32
  ! CHECK-DAG:  %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:  %[[a5:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[a6:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK:  %[[a7:.*]] = fir.convert %[[c4]] : (index) -> i32
    rslt = count(mask, dim=1)
  ! CHECK:  %{{.*}} = fir.call @_FortranACountDim(%[[a5]], %[[a6]], %[[c1_i32]], %[[a7]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
  ! CHECK:  %[[a10:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:  %[[a12:.*]] = fir.box_addr %[[a10]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:  fir.freemem %[[a12]]
  end subroutine
  
  ! CHECK-LABEL: test_count3
  ! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
  subroutine test_count3(rslt, mask)
    integer :: rslt
    logical :: mask(:)
  ! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[a1:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK:  %[[a3:.*]] = fir.convert %[[c0]] : (index) -> i32
    call bar(count(mask, kind=2))
  ! CHECK:  %[[a4:.*]] = fir.call @_FortranACount(%[[a1]], %{{.*}}, %{{.*}}, %[[a3]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
  ! CHECK:  %{{.*}} = fir.convert %[[a4]] : (i64) -> i16
  end subroutine

