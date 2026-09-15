// RUN: fir-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(loop-versioning),builtin.module(func.func(loop-versioning)),gpu.module(func.func(loop-versioning),builtin.module(func.func(loop-versioning))))' \
// RUN:   %s | FileCheck %s --enable-var-scope

// A non-builtin module does not establish the FIR-to-LLVM index contract.
// The nearest builtin module therefore supplies the p64 contract even when a
// surrounding GPU scope carries an unrelated p32 layout attribute. The i64
// lower bound makes that choice observable because a p32 contract rejects it.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:64:64:64:64"
} {
  func.func private @use_i32(!fir.ref<i32>)
  gpu.module @scope attributes {
    dlti.dl_spec = #dlti.dl_spec<>,
    llvm.data_layout = "e-p:32:32:32:32"
  } {
    func.func @gpu_local_layout_is_not_an_owner(
        %a: !fir.box<!fir.array<?xi32>>, %lower: i64) {
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %slice = fir.slice %lower, %c8, %c1
        : (i64, index, index) -> !fir.slice<1>
      fir.do_loop %i = %c1 to %c8 step %c1 {
        %address = fir.array_coor %a [%slice] %i
          : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index)
            -> !fir.ref<i32>
        fir.call @use_i32(%address) : (!fir.ref<i32>) -> ()
      }
      return
    }
  }
}

// CHECK-LABEL: func.func @gpu_local_layout_is_not_an_owner(
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

// -----

// A GPU scope does not make a nested builtin module unambiguous. The function
// may still be lowered under the inner or outer builtin module contract, so
// the initial slice path rejects it fail closed.
module attributes {
  dlti.dl_spec = #dlti.dl_spec<>,
  llvm.data_layout = "e-p:64:64:64:64"
} {
  gpu.module @scope attributes {
    dlti.dl_spec = #dlti.dl_spec<>,
    llvm.data_layout = "e-p:32:32:32:32"
  } {
    builtin.module @case attributes {dlti.dl_spec = #dlti.dl_spec<>} {
      func.func @nested_builtin_module_rejected(
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
}

// CHECK-LABEL: func.func @nested_builtin_module_rejected(
// CHECK-SAME: %[[A:[^:]+]]: !fir.box
// CHECK: %[[SLICE:.*]] = fir.slice
// CHECK-NOT: fir.box_dims %[[A]],
// CHECK-NOT: fir.if
// CHECK: fir.do_loop %[[I:[^ ]+]] =
// CHECK: fir.array_coor %[[A]] [%[[SLICE]]] %[[I]]
// CHECK-NOT: fir.if
// CHECK: return
