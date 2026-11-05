// Merging two blocks that differ only in a fir.slice or a fir.shape_shift
// passes that value as a block argument, leaving it with no defining op for
// cg-rewrite to read. Check that cg-rewrite takes the operands as block
// arguments instead and rebuilds the value in the block.

// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}),cg-rewrite)' | FileCheck %s
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}),cg-rewrite,fir-to-llvm-ir)' -o /dev/null

func.func private @use(!fir.box<!fir.array<?xf64>>)

// CHECK-LABEL: func.func @slice_block_argument
// CHECK:         cf.cond_br %{{.*}}, ^[[BB1:.*]](%{{.*}}, %{{.*}}, %{{.*}} : index, index, index), ^[[BB2:.*]]
// CHECK:       ^[[BB1]](%[[T0:.*]]: index, %[[T1:.*]]: index, %[[T2:.*]]: index):
// CHECK:         fircg.ext_embox %{{.*}}[%[[T0]], %[[T1]], %[[T2]]]
// CHECK:       ^[[BB2]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BB1]](%{{.*}}, %{{.*}}, %{{.*}} : index, index, index)
// CHECK-NOT:     !fir.slice
func.func @slice_block_argument(%cond1: i1, %cond2: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shift = fir.shape_shift %c1, %c4 : (index, index) -> !fir.shapeshift<1>
  %heap = fir.allocmem !fir.array<?xf64>, %c4
  %slice1 = fir.slice %c1, %c2, %c1 : (index, index, index) -> !fir.slice<1>
  %slice2 = fir.slice %c1, %c3, %c1 : (index, index, index) -> !fir.slice<1>
  cf.cond_br %cond1, ^bb1, ^between
^bb1:
  %emb1 = fir.embox %heap(%shift) [%slice1] : (!fir.heap<!fir.array<?xf64>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf64>>
  func.call @use(%emb1) : (!fir.box<!fir.array<?xf64>>) -> ()
  cf.br ^exit
^between:
  cf.cond_br %cond2, ^bb2, ^exit
^bb2:
  %emb2 = fir.embox %heap(%shift) [%slice2] : (!fir.heap<!fir.array<?xf64>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf64>>
  func.call @use(%emb2) : (!fir.box<!fir.array<?xf64>>) -> ()
  cf.br ^exit
^exit:
  fir.freemem %heap : !fir.heap<!fir.array<?xf64>>
  return
}

// A fir.shape_shift arriving as a block argument used to reach a dyn_cast on a
// null defining op.

// CHECK-LABEL: func.func @shapeshift_block_argument
// CHECK:         cf.cond_br %{{.*}}, ^[[SBB1:.*]](%{{.*}}, %{{.*}} : index, index), ^[[SBB2:.*]]
// CHECK:       ^[[SBB1]](%[[ORIGIN:.*]]: index, %[[EXTENT:.*]]: index):
// CHECK:         fircg.ext_embox %{{.*}}(%[[EXTENT]]) origin %[[ORIGIN]]
// CHECK-NOT:     !fir.shapeshift
func.func @shapeshift_block_argument(%cond1: i1, %cond2: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %heap = fir.allocmem !fir.array<?xf64>, %c4
  %shift1 = fir.shape_shift %c1, %c2 : (index, index) -> !fir.shapeshift<1>
  %shift2 = fir.shape_shift %c1, %c3 : (index, index) -> !fir.shapeshift<1>
  cf.cond_br %cond1, ^bb1, ^between
^bb1:
  %emb1 = fir.embox %heap(%shift1) : (!fir.heap<!fir.array<?xf64>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf64>>
  func.call @use(%emb1) : (!fir.box<!fir.array<?xf64>>) -> ()
  cf.br ^exit
^between:
  cf.cond_br %cond2, ^bb2, ^exit
^bb2:
  %emb2 = fir.embox %heap(%shift2) : (!fir.heap<!fir.array<?xf64>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf64>>
  func.call @use(%emb2) : (!fir.box<!fir.array<?xf64>>) -> ()
  cf.br ^exit
^exit:
  fir.freemem %heap : !fir.heap<!fir.array<?xf64>>
  return
}
