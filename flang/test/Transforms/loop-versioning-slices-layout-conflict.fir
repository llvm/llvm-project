// RUN: fir-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(loop-versioning),builtin.module(func.func(loop-versioning)))' \
// RUN:   --split-input-file %s | FileCheck %s --enable-var-scope
// RUN: fir-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(loop-versioning),builtin.module(func.func(loop-versioning)),cfg-conversion,cg-rewrite,fir-to-llvm-ir,reconcile-unrealized-casts)' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER \
// RUN:   --enable-var-scope

// A function in a nested builtin module can be lowered under either the inner
// or outer module contract. The initial slice path rejects this scope
// even when both modules currently carry matching attributes.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:64:64:64:64"
} {
  module @inner attributes {
    dlti.dl_spec = #dlti.dl_spec<>,
    llvm.data_layout = "e-p:64:64:64:64"
  } {
    func.func @nested_module_rejected(
        %a: !fir.box<!fir.array<?xi32>>) {
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %slice = fir.slice %c1, %c8, %c1
        : (index, index, index) -> !fir.slice<1>
      fir.do_loop %i = %c1 to %c8 step %c1 {
        %address = fir.array_coor %a [%slice] %i
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
      }
      return
    }
  }
}

// CHECK-LABEL: func.func @nested_module_rejected(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box
// CHECK: %[[SLICE:.*]] = fir.slice
// CHECK-NOT: fir.box_dims %[[A]],
// CHECK-NOT: fir.if
// CHECK: fir.do_loop %[[I:[^ ]+]] =
// CHECK: fir.array_coor %[[A]] [%[[SLICE]]] %[[I]]
// CHECK-NOT: fir.if
// CHECK: return

// LOWER-LABEL: llvm.func @nested_module_rejected(
// LOWER-NOT: llvm.icmp "eq"
// LOWER: llvm.getelementptr
// LOWER: llvm.return

// -----

// A top-level FIR module without an explicit LLVM layout uses the same
// default-64 contract as FIR-to-LLVM and remains eligible. The i64 lower bound
// would be rejected if the missing layout were incorrectly treated as p32.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>
} {
  func.func @default_layout(
      %a: !fir.box<!fir.array<?xi32>>, %lower: i64) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slice = fir.slice %lower, %c8, %c1
      : (i64, index, index) -> !fir.slice<1>
    fir.do_loop %i = %c1 to %c8 step %c1 {
      %address = fir.array_coor %a [%slice] %i
        : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
          -> !fir.ref<i32>
    }
    return
  }
}

// CHECK-LABEL: func.func @default_layout(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box
// CHECK: fir.box_dims %[[A]],
// CHECK: fir.if
// CHECK: %[[BYTE_BOX:.*]] = fir.convert %[[A]]
// CHECK-SAME: -> !fir.box<!fir.array<?xi8>>
// CHECK: %[[BYTES:.*]] = fir.box_addr %[[BYTE_BOX]]
// CHECK-SAME: -> !fir.ref<!fir.array<?xi8>>
// CHECK: fir.coordinate_of %[[BYTES]],
// CHECK-SAME: -> !fir.ref<i8>
// CHECK: } else {
// CHECK: fir.array_coor %[[A]]
// CHECK: return

// LOWER-LABEL: llvm.func @default_layout(
// LOWER: llvm.icmp "eq"
// LOWER: llvm.cond_br
// LOWER: llvm.getelementptr
// LOWER: llvm.return
