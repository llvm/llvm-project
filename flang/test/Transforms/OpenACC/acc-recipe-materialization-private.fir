// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

acc.private.recipe @privatization_ref_i64 : !fir.ref<i64> init {
^bb0(%arg0: !fir.ref<i64>):
  %0 = fir.alloca i64
  acc.yield %0 : !fir.ref<i64>
}

// CHECK-LABEL: func.func @private_i64
// CHECK: acc.loop control([[IV:%.+]] : i64)
// CHECK: [[ALLOC:%.+]] = fir.alloca i64
// CHECK: [[DECL:%.+]] = fir.declare [[ALLOC]] {uniq_name = "_private_arg0"}
// CHECK: fir.store [[IV]] to [[DECL]]

func.func @private_i64(%arg0 : !fir.ref<i64>) {
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %priv = acc.private varPtr(%arg0 : !fir.ref<i64>) recipe(@privatization_ref_i64) implicit(true) name("") -> !fir.ref<i64>
  acc.loop private(%priv : !fir.ref<i64>) control(%siv : i64) = (%c1_i32 : i32) to (%c16_i32 : i32) step (%c1_i32 : i32) {
    %priv_decl = fir.declare %priv {uniq_name = "_private_arg0"} : (!fir.ref<i64>) -> !fir.ref<i64>
    fir.store %siv to %priv_decl : !fir.ref<i64>
    acc.yield
  } independent
  return
}

// CHECK-LABEL: func.func @par_private_i64
// CHECK: acc.parallel {
// CHECK: [[ALLOC:%.+]] = fir.alloca i64
// CHECK: [[DECL:%.+]] = fir.declare [[ALLOC]] {uniq_name = "_private_arg0"}
// CHECK: acc.loop control([[IV:%.+]] : i64)
// CHECK: fir.store [[IV]] to [[DECL]]

func.func @par_private_i64(%arg0 : !fir.ref<i64>) {
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %priv = acc.private varPtr(%arg0 : !fir.ref<i64>) recipe(@privatization_ref_i64) implicit(true) name("") -> !fir.ref<i64>
  acc.parallel private(%priv : !fir.ref<i64>) {
    %priv_decl = fir.declare %priv {uniq_name = "_private_arg0"} : (!fir.ref<i64>) -> !fir.ref<i64>
    acc.loop control(%siv : i64) = (%c1_i32 : i32) to (%c16_i32 : i32) step (%c1_i32 : i32) {
      fir.store %siv to %priv_decl : !fir.ref<i64>
      acc.yield
    } independent
    acc.yield
  }
  return
}

acc.private.recipe @privatization_pointer : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
  %c4 = arith.constant 4 : index
  %alloc = fir.allocmem !fir.array<?xf32>, %c4
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %box = fir.embox %alloc(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %desc = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %box to %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  acc.yield %desc, %alloc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.heap<!fir.array<?xf32>>
} destroy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
     %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
     %owned: !fir.heap<!fir.array<?xf32>>):
  fir.freemem %owned : !fir.heap<!fir.array<?xf32>>
  acc.terminator
}

// CHECK-LABEL: func.func @private_pointer
// CHECK: %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>
// CHECK: %[[DESC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
// CHECK: fir.store %arg1 to %[[DESC]]
// CHECK-NOT: fir.box_addr
// CHECK: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xf32>>

func.func @private_pointer(
    %arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
    %arg1: !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
  %priv = acc.private varPtr(%arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) recipe(@privatization_pointer) name("p") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  acc.parallel private(%priv : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
    fir.store %arg1 to %priv : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    acc.yield
  }
  return
}
