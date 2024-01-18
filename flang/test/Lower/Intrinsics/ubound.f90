! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPubound_test(
subroutine ubound_test(a, dim, res)
  real, dimension(:, :) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_0:.*]] = fir.load
! CHECK:         %[[VAL_1:.*]] = fir.address_of(
! CHECK:         %[[VAL_2:.*]] = fir.convert
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_0]] : (i64) -> i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_1]]
! CHECK:         %[[VAL_5:.*]] = fir.call @_FortranASizeDim(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_6:.*]] = fir.address_of(
! CHECK:         %[[VAL_7:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (i64) -> i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_6]]
! CHECK:         %[[VAL_10:.*]] = fir.call @_FortranALboundDim(%[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %c1_i64 : i64
! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_5]] : i64

! CHECK:         fir.store %[[VAL_12]] to %{{.*}} : !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_2(
subroutine ubound_test_2(a, dim, res)
  real, dimension(2:, 3:) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_0:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK:         %[[VAL_1:.*]] = fir.address_of(
! CHECK:         %[[VAL_2:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_0]] : (i64) -> i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_1]]
! CHECK:         %[[VAL_5:.*]] = fir.call @_FortranASizeDim(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_6:.*]] = fir.address_of(
! CHECK:         %[[VAL_7:.*]] = fir.convert %{{.*}} : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (i64) -> i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_6]]
! CHECK:         %[[VAL_10:.*]] = fir.call @_FortranALboundDim(%[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %{{.*}} : i64
! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_5]] : i64
! CHECK:         fir.store %[[VAL_12]] to %{{.*}} : !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine

! CHECK-LABEL: func @_QPubound_test_3(
subroutine ubound_test_3(a, dim, res)
  real, dimension(10, 20, *) :: a
  integer(8):: dim, res
! CHECK:         %[[VAL_0:.*]] = fir.undefined index
! CHECK:         %[[VAL_1:.*]] = fir.shape %{{.*}}, %{{.*}}, %[[VAL_0]] : (index, index, index) -> !fir.shape<3>
! CHECK:         %[[VAL_2:.*]] = fir.embox %{{.*}}(%[[VAL_1]]) : (!fir.ref<!fir.array<10x20x?xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<10x20x?xf32>>
! CHECK:         %[[VAL_3:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.address_of(
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.array<10x20x?xf32>>) -> !fir.box<none>

! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (i64) -> i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]]
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranASizeDim(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_9:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.array<10x20x?xf32>>) -> !fir.box<!fir.array<10x20x?xf32>>
! CHECK:         %[[VAL_10:.*]] = fir.address_of(
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<10x20x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_3]]
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_10]]
! CHECK:         %[[VAL_14:.*]] = fir.call @_FortranALboundDim(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %{{.*}} : i64
! CHECK:         %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_8]] : i64
! CHECK:         fir.store %[[VAL_16]] to %{{.*}} : !fir.ref<i64>
  res = ubound(a, dim, 8)
end subroutine


! CHECK-LABEL: func @_QPubound_test_const_dim(
subroutine ubound_test_const_dim(array)
  real :: array(11:)
  integer :: res
! Should not call _FortranASizeDim when dim is compile time constant. But instead load from descriptor directly.
! CHECK:         %[[C0:.*]] = arith.constant 0 : index
! CHECK:         %[[DIMS:.*]]:3 = fir.box_dims %arg0, %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:         %{{.*}} = fir.convert %[[DIMS]]#1 : (index) -> i32
  res = ubound(array, 1)
end subroutine
