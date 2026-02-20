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

