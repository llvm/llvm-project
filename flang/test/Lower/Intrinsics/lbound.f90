! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s


! CHECK-LABEL: func.func @_QPlbound_test(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "dim"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "res"}) {
subroutine lbound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:         %[[DIM:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:         %[[RES:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:         %[[DIM_VAL:.*]] = fir.load %[[DIM]]#0 : !fir.ref<i64>
! CHECK:         %[[REBOX:.*]] = fir.rebox %[[A]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:         %[[A_NONE:.*]] = fir.convert %[[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[DIM_I32:.*]] = fir.convert %[[DIM_VAL]] : (i64) -> i32
! CHECK:         %[[RESULT:.*]] = fir.call @_FortranALboundDim(%[[A_NONE]], %[[DIM_I32]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         hlfir.assign %[[RESULT]] to %[[RES]]#0 : i64, !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func.func @_QPlbound_test_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "dim"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "res"}) {
subroutine lbound_test_2(a, dim, res)
  real, dimension(:, 2:) :: a
  integer(8):: dim, res
! CHECK:  %[[c1_i64:.*]] = arith.constant 1 : i64
! CHECK:  %[[IDX1:.*]] = fir.convert %[[c1_i64]] : (i64) -> index
! CHECK:  %[[c2_i64:.*]] = arith.constant 2 : i64
! CHECK:  %[[IDX2:.*]] = fir.convert %[[c2_i64]] : (i64) -> index
! CHECK:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]](%{{.*}})
! CHECK:  %[[DIM:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[RES:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:  %[[DIM_VAL:.*]] = fir.load %[[DIM]]#0 : !fir.ref<i64>
! CHECK:  %[[SHIFT:.*]] = fir.shift %[[IDX1]], %[[IDX2]] : (index, index) -> !fir.shift<2>
! CHECK:  %[[REBOX:.*]] = fir.rebox %[[A]]#1(%[[SHIFT]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:  %[[A_NONE:.*]] = fir.convert %[[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:  %[[DIM_I32:.*]] = fir.convert %[[DIM_VAL]] : (i64) -> i32
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranALboundDim(%[[A_NONE]], %[[DIM_I32]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         hlfir.assign %[[RESULT]] to %[[RES]]#0 : i64, !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

subroutine lbound_test_3(a, dim, res)
  real, dimension(2:10, 3:*) :: a
  integer(8):: dim, res
! CHECK:  %[[VAL_0:.*]] = fir.assumed_size_extent : index
! CHECK:  %[[DIM:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlbound_test_3Edim"}
! CHECK:  %[[DIM_VAL:.*]] = fir.load %[[DIM]]#0 : !fir.ref<i64>
! CHECK:  %[[SHAPESHIFT:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_0]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[EMBOX:.*]] = fir.embox %{{.*}}(%[[SHAPESHIFT]]) : (!fir.ref<!fir.array<9x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<9x?xf32>>
! CHECK:         %[[A_NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<9x?xf32>>) -> !fir.box<none>
! CHECK:         %[[DIM_I32:.*]] = fir.convert %[[DIM_VAL]] : (i64) -> i32
! CHECK:         %[[RESULT:.*]] = fir.call @_FortranALboundDim(%[[A_NONE]], %[[DIM_I32]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         hlfir.assign %[[RESULT]] to %{{.*}}#0 : i64, !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func.func @_QPlbound_test_4(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "dim"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "l1"},
! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<i64> {fir.bindc_name = "u1"},
! CHECK-SAME:  %[[VAL_4:.*]]: !fir.ref<i64> {fir.bindc_name = "l2"},
! CHECK-SAME:  %[[VAL_5:.*]]: !fir.ref<i64> {fir.bindc_name = "u2"}) {
subroutine lbound_test_4(a, dim, l1, u1, l2, u2)
  integer(8):: dim, l1, u1, l2, u2
! CHECK:  %[[TMP:.*]] = fir.alloca !fir.array<2xi32>
! CHECK-DAG:  %[[L1:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {uniq_name = "_QFlbound_test_4El1"}
! CHECK-DAG:  %[[L2:.*]]:2 = hlfir.declare %[[VAL_4]] {{.*}} {uniq_name = "_QFlbound_test_4El2"}
  real, dimension(l1:u1, l2:u2) :: a
! BeginExternalListOutput
! CHECK:  %[[L1_VAL:.*]] = fir.load %[[L1]]#0 : !fir.ref<i64>
! CHECK:  %[[L1_IDX:.*]] = fir.convert %[[L1_VAL]] : (i64) -> index
! CHECK:  %[[L2_VAL:.*]] = fir.load %[[L2]]#0 : !fir.ref<i64>
! CHECK:  %[[L2_IDX:.*]] = fir.convert %[[L2_VAL]] : (i64) -> index
! CHECK:  %[[c1_i32:.*]] = arith.constant 1 : i32
! CHECK:  %[[c0:.*]] = arith.constant 0 : index
! CHECK:  %[[EQ1:.*]] = arith.cmpi eq, %[[SIZE1:.*]], %[[c0]] : index
! CHECK:  %[[L1_AS_IDX:.*]] = fir.convert %[[c1_i32]] : (i32) -> index
! CHECK:  %[[LB1:.*]] = arith.select %[[EQ1]], %[[L1_AS_IDX]], %[[L1_IDX]] : index
! CHECK:  %[[LB1_I32:.*]] = fir.convert %[[LB1]] : (index) -> i32
! CHECK:  %[[c0_idx:.*]] = arith.constant 0 : index
! CHECK:  %[[COORD1:.*]] = fir.coordinate_of %[[TMP]], %[[c0_idx]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  fir.store %[[LB1_I32]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:  %[[EQ2:.*]] = arith.cmpi eq, %[[SIZE2:.*]], %[[c0]] : index
! CHECK:  %[[LB2:.*]] = arith.select %[[EQ2]], %{{.*}}, %[[L2_IDX]] : index
! CHECK:  %[[LB2_I32:.*]] = fir.convert %[[LB2]] : (index) -> i32
! CHECK:  %[[c1_idx:.*]] = arith.constant 1 : index
! CHECK:  %[[COORD2:.*]] = fir.coordinate_of %[[TMP]], %[[c1_idx]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  fir.store %[[LB2_I32]] to %[[COORD2]] : !fir.ref<i32>
! CHECK:  %[[c2:.*]] = arith.constant 2 : index
! CHECK:  %[[SHAPE:.*]] = fir.shape %[[c2]] : (index) -> !fir.shape<1>
! CHECK:  %[[TMPDECL:.*]]:2 = hlfir.declare %[[TMP]](%[[SHAPE]]) {uniq_name = ".tmp.intrinsic_result"}
! CHECK:  %[[EXPR:.*]] = hlfir.as_expr %[[TMPDECL]]#0 move %{{.*}} : (!fir.ref<!fir.array<2xi32>>, i1) -> !hlfir.expr<2xi32>
! CHECK:  %[[ASSOC:.*]]:3 = hlfir.associate %[[EXPR]](%[[SHAPE]]) {adapt.valuebyref}
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[ASSOC]]#0(%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:  %[[NONE:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, lbound(a, kind=4)
end subroutine
