! Check that a box is created instead of a temp to write to a char array.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine io_char_array
  character(12) :: r(2) = 'badbadbadbad'
  write(r(1:2),8)
  print *, r
1 format((1X,A))
8 format("new"/"data")
end subroutine

! CHECK-LABEL: func.func @_QPio_char_array()
! CHECK: %[[R:.*]] = fir.address_of(@_QFio_char_arrayEr) : !fir.ref<!fir.array<2x!fir.char<1,12>>>
! CHECK: %[[C2_SHAPE:.*]] = arith.constant 2 : index
! CHECK: %[[C1_I64_0:.*]] = arith.constant 1 : i64
! CHECK: %[[C1_IDX_0:.*]] = fir.convert %[[C1_I64_0]] : (i64) -> index
! CHECK: %[[C1_I64_1:.*]] = arith.constant 1 : i64
! CHECK: %[[C1_IDX_1:.*]] = fir.convert %[[C1_I64_1]] : (i64) -> index
! CHECK: %[[C2_I64:.*]] = arith.constant 2 : i64
! CHECK: %[[C2_IDX:.*]] = fir.convert %[[C2_I64]] : (i64) -> index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2_SHAPE]] : (index) -> !fir.shape<1>
! CHECK: %[[SLICE:.*]] = fir.slice %[[C1_IDX_0]], %[[C2_IDX]], %[[C1_IDX_1]] : (index, index, index) -> !fir.slice<1>
! CHECK: %[[BOX_R:.*]] = fir.embox %[[R]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<2x!fir.char<1,12>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2x!fir.char<1,12>>>
! CHECK: %[[BOX_R_NONE:.*]] = fir.convert %[[BOX_R]] : (!fir.box<!fir.array<2x!fir.char<1,12>>>) -> !fir.box<none>
! CHECK: %[[DATA:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,14>>
! CHECK: %[[DATA_PTR:.*]] = fir.convert %8 : (!fir.ref<!fir.char<1,14>>) -> !fir.ref<i8>
! CHECK: %{{.*}} = fir.call @_FortranAioBeginInternalArrayFormattedOutput(%[[BOX_R_NONE]], %[[DATA_PTR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i64, !fir.box<none>, !fir.ref<!fir.llvm_ptr<i8>>, i64, !fir.ref<i8>, i32) -> !fir.ref<i8>
