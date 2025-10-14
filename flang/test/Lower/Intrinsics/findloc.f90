! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfindloc_test_1d(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<1xi32>
function findloc_test_1d(a, v)
  integer :: a(:)
  integer :: v
  integer, dimension(1) :: findloc_test_1d
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_1d = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end function findloc_test_1d

! CHECK-LABEL: func @_QPfindloc_test_2d(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<2xi32>
function findloc_test_2d(a, v)
  integer :: a(:,:)
  integer :: v
  integer, dimension(2) :: findloc_test_2d
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_2d = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end function findloc_test_2d

! CHECK-LABEL: func @_QPfindloc_test_byval(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: i32{{.*}}) -> !fir.array<2xi32>
function findloc_test_byval(a, v)
  integer :: a(:,:)
  integer, value :: v
  integer, dimension(2) :: findloc_test_byval
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca i32
  ! CHECK-DAG: fir.store %[[arg1]] to %[[a1]] : !fir.ref<i32>
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[a1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_byval = findloc(a, v)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end function findloc_test_byval

! CHECK-LABEL: func @_QPfindloc_test_back_true(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) -> !fir.array<2xi32>
function findloc_test_back_true(a, v)
  integer :: a(:,:)
  integer :: v
  integer, dimension(2) :: findloc_test_back_true
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[true:.*]] = arith.constant true
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  findloc_test_back_true = findloc(a, v, back=.true.)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %true) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end function findloc_test_back_true

! CHECK-LABEL: func @_QPfindloc_test_back(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> !fir.array<2xi32>
function findloc_test_back(a, v, back)
  integer :: a(:,:)
  integer :: v
  logical :: back
  integer, dimension(2) :: findloc_test_back
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[arg2]] : !fir.ref<!fir.logical<4>>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  ! CHECK-DAG: %[[back:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
  findloc_test_back = findloc(a, v, back=back)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %[[back]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end function findloc_test_back

! CHECK-LABEL: func @_QPfindloc_test_dim(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_dim(a, v, res)
  integer :: a(:,:)
  integer :: v
  integer :: res(:)
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, dim=1)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[res]], %[[arr]], %[[val]], %[[kind]], %[[c1]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end subroutine findloc_test_dim

! CHECK-LABEL: func @_QPfindloc_test_dim_unknown(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<i32>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_dim_unknown(a, v, dim, res)
  integer :: a(:,:)
  integer :: v
  integer :: dim
  integer :: res(:)
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[dim:.*]] = fir.load %[[arg2]] : !fir.ref<i32>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, dim=dim)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[res]], %[[arr]], %[[val]], %[[kind]], %[[dim]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end subroutine findloc_test_dim_unknown

! CHECK-LABEL: func @_QPfindloc_test_kind(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?xi64>>{{.*}})
subroutine findloc_test_kind(a, v, res)
  integer :: a(:,:)
  integer :: v
  integer(8) :: res(:)
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.absent !fir.box<i1>
  ! CHECK-DAG: %[[kind:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<i1>) -> !fir.box<none>
  res = findloc(a, v, kind=8)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi64>>>) -> !fir.heap<!fir.array<?xi64>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi64>>
end subroutine findloc_test_kind

! CHECK-LABEL: func @_QPfindloc_test_non_scalar_mask(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}
subroutine findloc_test_non_scalar_mask(a, v, mask, res)
  integer :: a(:,:)
  integer :: v
  logical :: mask(:,:)
  integer :: res(:)
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, mask=mask)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
end subroutine findloc_test_non_scalar_mask

! CHECK-LABEL: func @_QPfindloc_test_scalar_mask(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine findloc_test_scalar_mask(a, v, mask, res)
  integer :: a(:,:)
  integer :: v
  logical :: mask
  integer :: res(:)
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[m:.*]] = fir.embox %[[arg2]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[m]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK-DAG: %[[kind:.*]] = fir.convert %[[c4]] : (index) -> i32
  res = findloc(a, v, mask=mask)
  ! CHECK:  fir.call @_FortranAFindloc(%[[res]], %[[arr]], %[[val]], %[[kind]], %{{.*}}, %{{.*}}, %[[mask]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi32>>
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
  ! CHECK-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
  ! CHECK-DAG: %[[v:.*]] = fir.embox %[[arg1]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[arg4]] : !fir.ref<!fir.logical<4>>
  ! CHECK-DAG: %[[kind:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[dim:.*]] = fir.load %[[arg2]] : !fir.ref<i32>
  ! CHECK-DAG: %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[val:.*]] = fir.convert %[[v]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG: %[[mask:.*]] = fir.convert %[[arg3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[back:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
  res = findloc(a, v, dim=dim, mask=mask, kind=8, back=back)
  ! CHECK:  fir.call @_FortranAFindlocDim(%[[res]], %[[arr]], %[[val]], %[[kind]], %[[dim]], %{{.*}}, %{{.*}}, %[[mask]], %[[back]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
  ! CHECK: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xi64>>>) -> !fir.heap<!fir.array<?xi64>>
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xi64>>
end subroutine findloc_test_all
