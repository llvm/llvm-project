! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfindloc_test_1d(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<1xi32>
function findloc_test_1d(a, v)
  integer :: a(:)
  integer :: v
  integer, dimension(1) :: findloc_test_1d
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[res_alloca:.*]] = fir.alloca !fir.array<1xi32>
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloca]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_1d = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<1xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end function findloc_test_1d

! CHECK-LABEL: func @_QPfindloc_test_2d(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<2xi32>
function findloc_test_2d(a, v)
  integer :: a(:,:)
  integer :: v
  integer, dimension(2) :: findloc_test_2d
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[res_alloca:.*]] = fir.alloca !fir.array<2xi32>
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloca]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_2d = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end function findloc_test_2d

! CHECK-LABEL: func @_QPfindloc_test_byval(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: i32{{.*}}) -> !fir.array<2xi32>
function findloc_test_byval(a, v)
  integer :: a(:,:)
  integer, value :: v
  integer, dimension(2) :: findloc_test_byval
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[v_alloca:.*]] = fir.alloca i32
  ! CHECK-DAG: fir.store %[[arg1]] to %[[v_alloca]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[v_alloca]]
  ! CHECK-DAG: %[[res_alloca:.*]] = fir.alloca !fir.array<2xi32>
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloca]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_byval = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end function findloc_test_byval

! CHECK-LABEL: func @_QPfindloc_test_back_true(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<2xi32>
function findloc_test_back_true(a, v)
  integer :: a(:,:)
  integer :: v
  integer, dimension(2) :: findloc_test_back_true
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[res_alloca:.*]] = fir.alloca !fir.array<2xi32>
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloca]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[true:.*]] = arith.constant true
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_back_true = findloc(a, v, back=.true.)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[true]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end function findloc_test_back_true

! CHECK-LABEL: func @_QPfindloc_test_back(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> !fir.array<2xi32>
function findloc_test_back(a, v, back)
  integer :: a(:,:)
  integer :: v
  logical :: back
  integer, dimension(2) :: findloc_test_back
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[back_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[res_alloca:.*]] = fir.alloca !fir.array<2xi32>
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloca]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[back_decl]]#0 : !fir.ref<!fir.logical<4>>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  ! CHECK-DAG: %[[back_i1:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
  findloc_test_back = findloc(a, v, back=back)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[back_i1]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end function findloc_test_back

! CHECK-LABEL: func @_QPfindloc_test_dim(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_dim(a, v, res)
  integer :: a(:,:)
  integer :: v
  integer :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, dim=1)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %[[c1_i32]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end subroutine findloc_test_dim

! CHECK-LABEL: func @_QPfindloc_test_dim_unknown(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<i32>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_dim_unknown(a, v, dim, res)
  integer :: a(:,:)
  integer :: v
  integer :: dim
  integer :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[dim_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[arg3]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[dim_val:.*]] = fir.load %[[dim_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, dim=dim)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %[[dim_val]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end subroutine findloc_test_dim_unknown

! CHECK-LABEL: func @_QPfindloc_test_kind(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?xi64>>{{.*}})
subroutine findloc_test_kind(a, v, res)
  integer :: a(:,:)
  integer :: v
  integer(8) :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[kind:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask]] : (!fir.box<i1>) -> !fir.box<none>
  res = findloc(a, v, kind=8)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi64>>>) -> !fir.heap<!fir.array<?xi64>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi64>, !fir.box<!fir.array<?xi64>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi64>
end subroutine findloc_test_kind

! CHECK-LABEL: func @_QPfindloc_test_non_scalar_mask(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}
subroutine findloc_test_non_scalar_mask(a, v, mask, res)
  integer :: a(:,:)
  integer :: v
  logical :: mask(:,:)
  integer :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %{{.*}}
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask_decl]]#1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, mask=mask)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end subroutine findloc_test_non_scalar_mask

! CHECK-LABEL: func @_QPfindloc_test_scalar_mask(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_scalar_mask(a, v, mask, res)
  integer :: a(:,:)
  integer :: v
  logical :: mask
  integer :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[arg3]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[mask_box:.*]] = fir.embox %[[mask_decl]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask_box]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, mask=mask)
  ! CHECK:  fir.call @_FortranAFindloc(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[false]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi32>
end subroutine findloc_test_scalar_mask

! CHECK-LABEL: func @_QPfindloc_test_all(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<i32>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[arg5:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}
subroutine findloc_test_all(a, v, dim, mask, back, res)
  integer :: a(:,:)
  integer :: v
  integer :: dim
  logical :: mask(:,:)
  logical :: back
  integer(8) :: res(:)
  ! CHECK-DAG: %[[a_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK-DAG: %[[back_decl:.*]]:2 = hlfir.declare %[[arg4]]
  ! CHECK-DAG: %[[dim_decl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg3]]
  ! CHECK-DAG: %[[res_decl:.*]]:2 = hlfir.declare %[[arg5]]
  ! CHECK-DAG: %[[v_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[box_alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
  ! CHECK-DAG: %[[kind:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[dim_val:.*]] = fir.load %[[dim_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[back_val:.*]] = fir.load %[[back_decl]]#0 : !fir.ref<!fir.logical<4>>
  ! CHECK-DAG: %[[v_box:.*]] = fir.embox %[[v_decl]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[box_none:.*]] = fir.convert %[[box_alloc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[a_none:.*]] = fir.convert %[[a_decl]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[v_none:.*]] = fir.convert %[[v_box]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask_none:.*]] = fir.convert %[[mask_decl]]#1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[back_i1:.*]] = fir.convert %[[back_val]] : (!fir.logical<4>) -> i1
  res = findloc(a, v, dim=dim, mask=mask, kind=8, back=back)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[box_none]], %[[a_none]], %[[v_none]], %[[kind]], %[[dim_val]], %{{.*}}, %{{.*}}, %[[mask_none]], %[[back_i1]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box_load:.*]] = fir.load %[[box_alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[box_addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.heap<!fir.array<?xi64>>>) -> !fir.heap<!fir.array<?xi64>>
  ! CHECK: %[[tmp_decl:.*]]:2 = hlfir.declare %[[box_addr]]
  ! CHECK: %[[expr:.*]] = hlfir.as_expr %[[tmp_decl]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
  ! CHECK: hlfir.assign %[[expr]] to %[[res_decl]]#0 : !hlfir.expr<?xi64>, !fir.box<!fir.array<?xi64>>
  ! CHECK: hlfir.destroy %[[expr]] : !hlfir.expr<?xi64>
end subroutine findloc_test_all
