! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPminloc_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>
subroutine minloc_test(arr,res)
    integer :: arr(:)
    integer :: res(:)
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[a1:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
  ! CHECK-DAG: %[[a10:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
    res = minloc(arr)
  ! CHECK: fir.call @_FortranAMinlocInteger4(%[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}, %[[a10]], %false) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK-DAG: %[[a12:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG: %[[a14:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK-DAG: fir.freemem %[[a14]]
  end subroutine
  
  ! CHECK-LABEL: func @_QPminloc_test2(
  ! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg2:.*]]: !fir.ref<i32>
  subroutine minloc_test2(arr,res,d)
    integer :: arr(:)
    integer :: res(:)
    integer :: d
  ! CHECK-DAG:  %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK-DAG:  %[[a1:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG:  %[[a2:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG:  %[[a7:.*]] = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
  ! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
    res = minloc(arr, dim=d)
  ! CHECK:  fir.call @_FortranAMinlocDim(%[[a6]], %[[a7]], %[[a8]], %[[a1]], %{{.*}}, %{{.*}}, %[[a10]], %false) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK:  %[[a12:.*]] = fir.load %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK:  %[[a13:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK:  fir.freemem %[[a13]]
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_minloc_optional_scalar_mask(
  ! CHECK-SAME:  %[[VAL_0:[^:]+]]: !fir.ref<!fir.logical<4>>
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>
  subroutine test_minloc_optional_scalar_mask(mask, back, array)
    integer :: array(:)
    logical, optional :: mask
    logical, optional :: back
    print *, minloc(array, mask=mask, back=back)
  ! CHECK:  %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK:  %[[VAL_11:.*]] = fir.absent !fir.box<!fir.logical<4>>
  ! CHECK:  %[[VAL_12:.*]] = arith.select %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : !fir.box<!fir.logical<4>>
  ! CHECK:  %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (!fir.logical<4>) {
    ! CHECK:  %[[VAL_15:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
    ! CHECK:  fir.result %[[VAL_15]] : !fir.logical<4>
  ! CHECK:  } else {
    ! CHECK:  %[[VAL_16:.*]] = arith.constant false
    ! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i1) -> !fir.logical<4>
    ! CHECK:  fir.result %[[VAL_17]] : !fir.logical<4>
  ! CHECK:  }
  ! CHECK:  %[[VAL_29:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_30:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
  ! CHECK:  fir.call @_FortranAMinlocInteger4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_29]], %[[VAL_30]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_minloc_optional_array_mask(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>
  subroutine test_minloc_optional_array_mask(mask, back, array)
    integer :: array(:)
    logical, optional :: mask(:)
    logical, optional :: back
    print *, minloc(array, mask=mask, back=back)
  ! CHECK:  %[[VAL_9:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[VAL_10:.*]] = fir.if %[[VAL_9]] -> (!fir.logical<4>) {
    ! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
    ! CHECK:  fir.result %[[VAL_11]] : !fir.logical<4>
  ! CHECK:  } else {
    ! CHECK:  %[[VAL_12:.*]] = arith.constant false
    ! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>
    ! CHECK:  fir.result %[[VAL_13]] : !fir.logical<4>
  ! CHECK:  }
  ! CHECK:  %[[VAL_25:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_10]] : (!fir.logical<4>) -> i1
  ! CHECK:  fir.call @_FortranAMinlocInteger4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_25]], %[[VAL_26]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  end subroutine
