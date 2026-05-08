! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  integer :: dim = 1
  integer :: iSize
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
! CHECK:         %[[VAL_SS:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_SS]]) {uniq_name = "_QFsize_testEa"}
! CHECK:         %[[DIM_ADDR:.*]] = fir.address_of(@_QFsize_testEdim) : !fir.ref<i32>
! CHECK:         %[[DIM_DECL:.*]]:2 = hlfir.declare %[[DIM_ADDR]] {uniq_name = "_QFsize_testEdim"}
! CHECK:         %[[ISIZE_ALLOC:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[ISIZE_DECL:.*]]:2 = hlfir.declare %[[ISIZE_ALLOC]] {uniq_name = "_QFsize_testEisize"}
! CHECK:         %[[SLICE_RESULT:.*]] = hlfir.designate %[[A_DECL]]#0 (%{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}}
! CHECK:         %[[DIM_PTR:.*]] = fir.convert %[[DIM_DECL]]#0 : (!fir.ref<i32>) -> i64
! CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:         %[[IS_ABSENT:.*]] = arith.cmpi eq, %[[DIM_PTR]], %[[C0_I64]] : i64
! CHECK:         %[[SIZE_RESULT:.*]] = fir.if %[[IS_ABSENT]] -> (i64) {
! CHECK:           %[[BOX1:.*]] = fir.convert %[[SLICE_RESULT]] : ({{.*}}) -> !fir.box<none>
! CHECK:           %[[SZ1:.*]] = fir.call @_FortranASize(%[[BOX1]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ1]] : i64
! CHECK:         } else {
! CHECK:           %[[DIM_VAL:.*]] = fir.load %[[DIM_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[BOX2:.*]] = fir.convert %[[SLICE_RESULT]] : ({{.*}}) -> !fir.box<none>
! CHECK:           %[[SZ2:.*]] = fir.call @_FortranASizeDim(%[[BOX2]], %[[DIM_VAL]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ2]] : i64
! CHECK:         }
! CHECK:         %[[SIZE_I32:.*]] = fir.convert %[[SIZE_RESULT]] : (i64) -> i32
! CHECK:         hlfir.assign %[[SIZE_I32]] to %[[ISIZE_DECL]]#0 : i32, !fir.ref<i32>
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
! CHECK-DAG:     %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:     %[[DIM_DECL:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK-DAG:     %[[ISIZE_DECL:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:         %[[DIM_I64:.*]] = fir.convert %[[DIM_DECL]]#0 : (!fir.ref<i32>) -> i64
! CHECK:         %[[C0:.*]] = arith.constant 0 : i64
! CHECK:         %[[IS_ABSENT:.*]] = arith.cmpi eq, %[[DIM_I64]], %[[C0]] : i64
! CHECK:         %[[RESULT:.*]] = fir.if %[[IS_ABSENT]] -> (i64) {
! CHECK:           %[[BOX1:.*]] = fir.convert %[[ARRAY_DECL]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[SZ1:.*]] = fir.call @_FortranASize(%[[BOX1]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ1]] : i64
! CHECK:         } else {
! CHECK:           %[[DIM_VAL:.*]] = fir.load %[[DIM_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[BOX2:.*]] = fir.convert %[[ARRAY_DECL]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[SZ2:.*]] = fir.call @_FortranASizeDim(%[[BOX2]], %[[DIM_VAL]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ2]] : i64
! CHECK:         }
! CHECK:         hlfir.assign %[[RESULT]] to %[[ISIZE_DECL]]#0 : i64, !fir.ref<i64>
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
! CHECK-DAG:     %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:     %[[DIM_DECL:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK-DAG:     %[[ISIZE_DECL:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:         %[[DIM_BOX:.*]] = fir.load %[[DIM_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[DIM_PTR:.*]] = fir.box_addr %[[DIM_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[DIM_I64:.*]] = fir.convert %[[DIM_PTR]] : (!fir.ptr<i32>) -> i64
! CHECK:         %[[C0:.*]] = arith.constant 0 : i64
! CHECK:         %[[IS_ABSENT:.*]] = arith.cmpi eq, %[[DIM_I64]], %[[C0]] : i64
! CHECK:         %[[RESULT:.*]] = fir.if %[[IS_ABSENT]] -> (i64) {
! CHECK:           %[[BOX1:.*]] = fir.convert %[[ARRAY_DECL]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[SZ1:.*]] = fir.call @_FortranASize(%[[BOX1]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ1]] : i64
! CHECK:         } else {
! CHECK:           %[[DIM_VAL:.*]] = fir.load %[[DIM_PTR]] : !fir.ptr<i32>
! CHECK:           %[[BOX2:.*]] = fir.convert %[[ARRAY_DECL]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[SZ2:.*]] = fir.call @_FortranASizeDim(%[[BOX2]], %[[DIM_VAL]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, i32, !fir.ref<i8>, i32) -> i64
! CHECK:           fir.result %[[SZ2]] : i64
! CHECK:         }
! CHECK:         hlfir.assign %[[RESULT]] to %[[ISIZE_DECL]]#0 : i64, !fir.ref<i64>
end subroutine
