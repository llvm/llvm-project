! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPminval_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32
integer function minval_test(a)
integer :: a(:)
! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[a2:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a4:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[a6:.*]] = fir.convert %[[c0]] : (index) -> i32
! CHECK:  %[[a7:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
minval_test = minval(a)
! CHECK:  %{{.*}} = fir.call @_FortranAMinvalInteger4(%[[a4]], %{{.*}}, %{{.*}}, %[[a6]], %[[a7]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func @_QPminval_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1>>{{.*}}, %[[arg1:.*]]: index{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.char<1>>>{{.*}}) -> !fir.boxchar<1>
character function minval_test2(a)
character :: a(:)
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a5:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[a6:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?x!fir.char<1>>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
minval_test2 = minval(a)
! CHECK:  %{{.*}} = fir.call @_FortranAMinvalCharacter(%[[a5]], %[[a6]], %{{.*}}, %{{.*}}, %[[a8]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32, !fir.box<none>) -> none
end function

! CHECK-LABEL: func @_QPminval_test3(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine minval_test3(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK-DAG:  %[[c2_i32:.*]] = arith.constant 2 : i32
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
r = minval(a,dim=2)
! CHECK:  %{{.*}} = fir.call @_FortranAMinvalDim(%[[a6]], %[[a7]], %[[c2_i32]], %{{.*}}, %{{.*}}, %[[a9]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> none
! CHECK:  %[[a11:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:  %[[a13:.*]] = fir.box_addr %[[a11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG:  fir.freemem %[[a13]]
end subroutine

! CHECK-LABEL: func @_QPtest_minval_optional_scalar_mask(
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>
subroutine test_minval_optional_scalar_mask(mask, array)
integer :: array(:)
logical, optional :: mask
print *, minval(array, mask)
! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
! CHECK:  %[[VAL_9:.*]] = fir.absent !fir.box<!fir.logical<4>>
! CHECK:  %[[VAL_10:.*]] = arith.select %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : !fir.box<!fir.logical<4>>
! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAMinvalInteger4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_17]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end subroutine

! CHECK-LABEL: func @_QPtest_minval_optional_array_mask(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_minval_optional_array_mask(mask, array)
integer :: array(:)
logical, optional :: mask(:)
print *, minval(array, mask)
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAMinvalInteger4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_13]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end subroutine
