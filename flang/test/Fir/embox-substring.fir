// RUN: fir-opt -o - -cg-rewrite --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco -o - -cg-rewrite --fir-to-llvm-ir %s | FileCheck %s

// CHECK-LABEL: llvm.func @embox_index_substr(
// CHECK-NOT: NULL_VALUE
// CHECK-NOT: NULL_TYPE
func.func @embox_index_substr(%addr : !fir.ref<!fir.array<?x!fir.char<1,2>>>) {
  %0 = arith.constant 0 : index
  %1 = fir.shape_shift %0, %0 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.slice %0, %0, %0 substr %0, %0: (index, index, index, index, index) -> !fir.slice<1>
  %3 = fir.embox %addr (%1) [%2] : (!fir.ref<!fir.array<?x!fir.char<1,2>>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  return
}

// CHARACTER(*) :: C(2)
// CALL DUMP(C(:)(1:1))
// Test that the resulting stride is based on the input length, not the substring one.
func.func @substring_dyn_base(%base_addr: !fir.ref<!fir.array<2x!fir.char<1,?>>>, %base_len: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %2 = fircg.ext_embox %base_addr(%c2)[%c1, %c2, %c1] substr %c0, %c1 typeparams %base_len : (!fir.ref<!fir.array<2x!fir.char<1,?>>>, index, index, index, index, index, index, index) -> !fir.box<!fir.array<2x!fir.char<1>>>
  fir.call @dump(%2) : (!fir.box<!fir.array<2x!fir.char<1>>>) -> ()
  return
}
func.func private @dump(!fir.box<!fir.array<2x!fir.char<1>>>)

// CHECK-LABEL:   llvm.func @substring_dyn_base(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.getelementptr
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_28]][1] : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_30:.*]] = llvm.ptrtoint %[[VAL_29]] : !llvm.ptr<i8> to i64
// CHECK:           %[[VAL_31:.*]] = llvm.mul %[[VAL_30]], %[[VAL_1]]  : i64
// CHECK:           %[[VAL_42:.*]] = llvm.mul %[[VAL_31]], %[[VAL_5]]  : i64
// CHECK:           %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_42]], %{{.*}}[7, 0, 2] : !llvm.struct<(ptr<array<2 x array<1 x i8>>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
