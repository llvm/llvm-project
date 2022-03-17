! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s


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
! CHECK:         %[[VAL_6:.*]] = fir.call @_FortranALboundDim(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
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
! CHECK:  %[[VAL_9:.*]] = fir.call @_FortranALboundDim(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         fir.store %[[VAL_9]] to %arg2 : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine

! CHECK:  %[[VAL_0:.*]] = fir.undefined index
subroutine lbound_test_3(a, dim, res)
  real, dimension(2:10, 3:*) :: a
  integer(8):: dim, res
! CHECK:  %[[VAL_1:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_0]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_3:.*]] = fir.embox %arg0(%[[VAL_2]]) : (!fir.ref<!fir.array<9x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<9x?xf32>>
! CHECK:         %[[VAL_4:.*]] = fir.address_of(
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.array<9x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (i64) -> i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]]
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranALboundDim(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         fir.store %[[VAL_8]] to %arg2 : !fir.ref<i64>
  res = lbound(a, dim, 8)
end subroutine
