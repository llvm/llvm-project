// RUN: fir-opt --stack-arrays %s | FileCheck %s

// Test that an allocmem inside a fir.if is not hoisted past a
// stackrestore when the size operand is shared across scopes.

func.func @nested_if_scope(%arg0: index, %arg1: i1) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %size = arith.addi %arg0, %arg0 : index

  %sp1 = llvm.intr.stacksave : !llvm.ptr
  %mem1 = fir.allocmem !fir.array<?xi32>, %size
  %ref1 = fir.convert %mem1 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %elt1 = fir.coordinate_of %ref1, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  fir.store %c0_i32 to %elt1 : !fir.ref<i32>
  fir.freemem %mem1 : !fir.heap<!fir.array<?xi32>>
  llvm.intr.stackrestore %sp1 : !llvm.ptr

  fir.if %arg1 {
    %sp2 = llvm.intr.stacksave : !llvm.ptr
    %mem2 = fir.allocmem !fir.array<?xi32>, %size
    %ref2 = fir.convert %mem2 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %elt2 = fir.coordinate_of %ref2, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt2 : !fir.ref<i32>
    fir.freemem %mem2 : !fir.heap<!fir.array<?xi32>>
    llvm.intr.stackrestore %sp2 : !llvm.ptr
  }

  return
}

// Second alloca must stay inside the fir.if.
// CHECK-LABEL: func.func @nested_if_scope(
// CHECK:         %[[SIZE:.*]] = arith.addi
// CHECK:         fir.alloca !fir.array<?xi32>, %[[SIZE]]
// CHECK:         llvm.intr.stacksave
// CHECK:         llvm.intr.stackrestore
// CHECK:         fir.if
// CHECK:           llvm.intr.stacksave
// CHECK:           fir.alloca !fir.array<?xi32>, %[[SIZE]]
// CHECK:           llvm.intr.stackrestore
