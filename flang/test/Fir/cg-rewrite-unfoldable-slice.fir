// A slice that reaches fir.embox through an arith.select has no fir.slice to
// read it from, and unlike a block argument there is nothing to take apart.
// Report it: folding it away would describe the whole array rather than the
// section the slice names.

// RUN: not fir-opt %s -cg-rewrite 2>&1 | FileCheck %s

func.func private @use(!fir.box<!fir.array<?xf64>>)

// CHECK: 'fir.embox' op slice operand is not defined by a fir.slice
func.func @selected_slice(%cond: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shift = fir.shape_shift %c1, %c4 : (index, index) -> !fir.shapeshift<1>
  %heap = fir.allocmem !fir.array<?xf64>, %c4
  %slice1 = fir.slice %c1, %c2, %c1 : (index, index, index) -> !fir.slice<1>
  %slice2 = fir.slice %c1, %c3, %c1 : (index, index, index) -> !fir.slice<1>
  %slice = arith.select %cond, %slice1, %slice2 : !fir.slice<1>
  %emb = fir.embox %heap(%shift) [%slice] : (!fir.heap<!fir.array<?xf64>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf64>>
  func.call @use(%emb) : (!fir.box<!fir.array<?xf64>>) -> ()
  fir.freemem %heap : !fir.heap<!fir.array<?xf64>>
  return
}
