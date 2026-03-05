// Verify fir.array_coor lowering when the base is a block argument.
// This used to take a shortcut (convert the element ref result) which kept the
// fir.array_coor alive. We prefer converting the base to a memref and
// computing indices so that fir.array_coor can be erased when only used by
// load/store.
//
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

func.func @block_arg_memref(%arg0: !fir.ref<!fir.array<32xi32>>) {
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c32 : (index) -> !fir.shape<1>
  %elt = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<32xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %elt : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @block_arg_memref
// CHECK:       [[BASE:%.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<32xi32>>) -> memref<32xi32>
// CHECK:       [[ONE:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB:%.+]] = arith.subi %c1, [[ONE]] : index
// CHECK:       [[MUL:%.+]] = arith.muli [[SUB]], [[ONE]] : index
// CHECK:       [[SUB2:%.+]] = arith.subi [[ONE]], [[ONE]] : index
// CHECK:       [[IDX:%.+]] = arith.addi [[MUL]], [[SUB2]] : index
// CHECK:       memref.store {{%.+}}, [[BASE]][[[IDX]]] : memref<32xi32>
// CHECK-NOT:   fir.array_coor

// Verify fir.array_coor lowering when the base is a boxed SCF region block
// argument. The pass must materialize fir.box_addr
// first and must not emit an illegal direct box->memref fir.convert.
func.func @block_arg_boxed_array(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %loop_res = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg1 = %arg0) -> (!fir.box<!fir.array<?xi32>>) {
    %idxp1 = arith.addi %i, %c1 : index
    %val = arith.index_cast %idxp1 : index to i32
    %idx64 = arith.extsi %val : i32 to i64
    %elt = fir.array_coor %arg1 %idx64 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    fir.store %val to %elt : !fir.ref<i32>
    scf.yield %arg1 : !fir.box<!fir.array<?xi32>>
  }
  return
}

// CHECK-LABEL: func.func @block_arg_boxed_array
// CHECK:         scf.for {{.*}} iter_args([[BOXARG:%.+]] = %arg0) -> (!fir.box<!fir.array<?xi32>>) {
// CHECK:           [[BOXADDR:%.+]] = fir.box_addr [[BOXARG]]
// CHECK:           [[BASE:%.+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?xi32>>) -> memref<?xi32{{.*}}>
// CHECK-NOT:       fir.convert [[BOXARG]] : (!fir.box<!fir.array<?xi32>>) -> memref
// CHECK:           memref.store {{%.+}}, {{%.+}}[{{%.+}}] : memref<?xi32, strided<[?], offset: ?>>
// CHECK-NOT:       fir.array_coor

