! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPlbound_test(
! CHECK:  %[[VAL_0:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK:  %[[VAL_1:.*]] = fir.rebox %arg0 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
subroutine lbound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_2:.*]] = fir.address_of(
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (i64) -> i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_2]]
! CHECK:         %[[VAL_6:.*]] = fir.call @_FortranALboundDim(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         fir.store %[[VAL_6]] to %arg2 : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPlbound_test_2(
! CHECK:  %[[VAL_c1_i64:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_0:.*]] = fir.convert %[[VAL_c1_i64]] : (i64) -> index
! CHECK:  %[[VAL_c2_i64:.*]] = arith.constant 2 : i64
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_c2_i64]] : (i64) -> index
! CHECK:  %[[VAL_2:.*]] = fir.load %arg1 : !fir.ref<i64>
subroutine lbound_test_2(a, dim, res)
  real, dimension(:, 2:) :: a
  integer(8):: dim, res
! CHECK:  %[[VAL_3:.*]] = fir.shift %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shift<2>
! CHECK:  %[[VAL_4:.*]] = fir.rebox %arg0(%[[VAL_3]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_5:.*]] = fir.address_of(
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (i64) -> i32
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_5]]
! CHECK:  %[[VAL_9:.*]] = fir.call @_FortranALboundDim(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         fir.store %[[VAL_9]] to %arg2 : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

subroutine lbound_test_3(a, dim, res)
  real, dimension(2:10, 3:*) :: a
  integer(8):: dim, res
! CHECK:  %[[VAL_0:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_1:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_0]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_3:.*]] = fir.embox %arg0(%[[VAL_2]]) : (!fir.ref<!fir.array<9x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<9x?xf32>>
! CHECK:         %[[VAL_4:.*]] = fir.address_of(
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.array<9x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (i64) -> i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]]
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranALboundDim(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         fir.store %[[VAL_8]] to %arg2 : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPlbound_test_4(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "dim"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "l1"},
! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<i64> {fir.bindc_name = "u1"},
! CHECK-SAME:  %[[VAL_4:.*]]: !fir.ref<i64> {fir.bindc_name = "l2"},
! CHECK-SAME:  %[[VAL_5:.*]]: !fir.ref<i64> {fir.bindc_name = "u2"}) {
subroutine lbound_test_4(a, dim, l1, u1, l2, u2)
  integer(8):: dim, l1, u1, l2, u2
! CHECK:  %[[VAL_6:.*]] = fir.alloca !fir.array<2xi32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:  %[[VAL_17:.*]] = fir.load %[[VAL_4]] : !fir.ref<i64>
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
  real, dimension(l1:u1, l2:u2) :: a
! BeginExternalListOutput
! CHECK:  %[[VAL_32:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_33:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_34:.*]] = arith.cmpi eq, %[[VAL_16:.*]], %[[VAL_33]] : index
! CHECK:  %[[VAL_35:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
! CHECK:  %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_35]], %[[VAL_8]] : index
! CHECK:  %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (index) -> i32
! CHECK:  %[[VAL_38:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_39:.*]] = fir.coordinate_of %[[VAL_6]], %[[VAL_38]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  fir.store %[[VAL_37]] to %[[VAL_39]] : !fir.ref<i32>
! CHECK:  %[[VAL_40:.*]] = arith.cmpi eq, %[[VAL_26:.*]], %[[VAL_33]] : index
! CHECK:  %[[VAL_41:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
! CHECK:  %[[VAL_42:.*]] = arith.select %[[VAL_40]], %[[VAL_41]], %[[VAL_18]] : index
! CHECK:  %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (index) -> i32
! CHECK:  %[[VAL_44:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_6]], %[[VAL_44]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  fir.store %[[VAL_43]] to %[[VAL_45]] : !fir.ref<i32>
! CHECK:  %[[VAL_46:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_47:.*]] = fir.shape %[[VAL_46]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_48:.*]] = fir.embox %[[VAL_6]](%[[VAL_47]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:  %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[VAL_49]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, lbound(a, kind=4)
end subroutine
