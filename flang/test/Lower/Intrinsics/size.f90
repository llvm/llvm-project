! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  integer :: dim = 1
  integer :: iSize
! CHECK:         %[[VAL_c1:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_c10:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_cneg10:.*]] = arith.constant -10 : index
! CHECK:         %[[VAL_c21:.*]] = arith.constant 21 : index
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QFsize_testEdim) : !fir.ref<i32>
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[VAL_c2_i64:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_c2_i64]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_c1_i64]] : (i64) -> index
! CHECK:         %[[VAL_c5_i64:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_c5_i64]] : (i64) -> index
! CHECK:         %[[VAL_neg1_i64:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_neg1_i64]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64_0:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_c1_i64_0]] : (i64) -> index
! CHECK:         %[[VAL_c1_i64_1:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_c1_i64_1]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = fir.shape_shift %[[VAL_c1]], %[[VAL_c10]], %[[VAL_cneg10]], %[[VAL_c21]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_10:.*]] = fir.slice %[[VAL_3]], %[[VAL_5]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]], %[[VAL_7]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_0]](%[[VAL_9]]) [%[[VAL_10]]] : (!fir.ref<!fir.array<10x21xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<4x3xf32>>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[c0_i64:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_12]], %[[c0_i64]] : i64
! CHECK:         %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (i64) {
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<4x3xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranASize(%[[VAL_17]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_19]] : i64
! CHECK:         } else {
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<4x3xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranASizeDim(%[[VAL_18]], %[[VAL_16]], %{{.*}}, %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_20]] : i64
! CHECK:         }
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_14]] : (i64) -> i32
! CHECK:         fir.store %[[VAL_21]] to %[[VAL_2]] : !fir.ref<i32>
  iSize = size(a(2:5, -1:1), dim, 8)
end subroutine size_test

! CHECK-LABEL: func @_QPsize_optional_dim_1(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "dim", fir.optional},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "isize"}) {
subroutine size_optional_dim_1(array, dim, iSize)
  real, dimension(:,:) :: array
  integer, optional :: dim
  integer(8) :: iSize
  iSize = size(array, dim, 8)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:         %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (i64) {
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranASize(%[[VAL_9]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_11]] : i64
! CHECK:         } else {
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranASizeDim(%[[VAL_15]], %[[VAL_12]], %{{.*}}, %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_17]] : i64
! CHECK:         }
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<i64>
end subroutine

! CHECK-LABEL: func @_QPsize_optional_dim_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "dim"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "isize"}) {
subroutine size_optional_dim_2(array, dim, iSize)
  real, dimension(:,:) :: array
  integer, pointer :: dim
  integer(8) :: iSize
  iSize = size(array, dim, 8)
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<i32>) -> i64
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:         %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (i64) {
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.call @_FortranASize(%[[VAL_11]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_13]] : i64
! CHECK:         } else {
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_4]] : !fir.ptr<i32>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranASizeDim(%[[VAL_17]], %[[VAL_14]], %{{.*}}, %{{.*}}) : (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[VAL_19]] : i64
! CHECK:         }
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<i64>
end subroutine
