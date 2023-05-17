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
