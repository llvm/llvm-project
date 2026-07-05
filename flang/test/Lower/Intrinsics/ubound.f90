! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPubound_test(
subroutine ubound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK-DAG: %[[aDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_testEa"}
! CHECK-DAG: %[[dimDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_testEdim"}
! CHECK-DAG: %[[resDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_testEres"}
! CHECK: %[[dimVal:.*]] = fir.load %[[dimDecl]]#0 : !fir.ref<i64>
! CHECK: %[[size:.*]] = fir.call @_FortranASizeDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lbound:.*]] = fir.call @_FortranALboundDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lb_m1:.*]] = arith.subi %[[lbound]], {{.*}} : i64
! CHECK: %[[ub:.*]] = arith.addi %[[lb_m1]], %[[size]] : i64
! CHECK: hlfir.assign %[[ub]] to %[[resDecl]]#0 : i64, !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_2(
subroutine ubound_test_2(a, dim, res)
  real, dimension(2:, 3:) :: a
  integer(8):: dim, res
! CHECK-DAG: %[[aDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_test_2Ea"}
! CHECK-DAG: %[[dimDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_test_2Edim"}
! CHECK-DAG: %[[resDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_test_2Eres"}
! CHECK: %[[size:.*]] = fir.call @_FortranASizeDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lbound:.*]] = fir.call @_FortranALboundDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lb_m1:.*]] = arith.subi %[[lbound]], {{.*}} : i64
! CHECK: %[[ub:.*]] = arith.addi %[[lb_m1]], %[[size]] : i64
! CHECK: hlfir.assign %[[ub]] to %[[resDecl]]#0 : i64, !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_3(
subroutine ubound_test_3(a, dim, res)
  real, dimension(10, 20, *) :: a
  integer(8):: dim, res
! CHECK: %[[assumed:.*]] = fir.assumed_size_extent : index
! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}}, %[[assumed]] : (index, index, index) -> !fir.shape<3>
! CHECK: %[[aDecl:.*]]:2 = hlfir.declare {{.*}}(%[[shape]]) {{.*}}{uniq_name = "_QFubound_test_3Ea"}
! CHECK: %[[embox:.*]] = fir.embox %[[aDecl]]#1(%{{.*}}) : (!fir.ref<!fir.array<10x20x?xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<10x20x?xf32>>
! CHECK: %[[size:.*]] = fir.call @_FortranASizeDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lbound:.*]] = fir.call @_FortranALboundDim({{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK: %[[lb_m1:.*]] = arith.subi %[[lbound]], {{.*}} : i64
! CHECK: %[[ub:.*]] = arith.addi %[[lb_m1]], %[[size]] : i64
! CHECK: hlfir.assign %[[ub]] to {{.*}} : i64, !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine


! CHECK-LABEL: func @_QPubound_test_const_dim(
subroutine ubound_test_const_dim(array)
  real :: array(11:)
  integer :: res
! Should not call _FortranASizeDim when dim is compile time constant. But instead load from descriptor directly.
! CHECK: %[[arrayDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFubound_test_const_dimEarray"}
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[arrayDecl]]#1, %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %{{.*}} = fir.convert %[[DIMS]]#1 : (index) -> i32
  res = ubound(array, 1)
end subroutine
