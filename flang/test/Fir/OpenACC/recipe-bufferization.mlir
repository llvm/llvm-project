// RUN: fir-opt %s --fir-acc-recipe-bufferization -split-input-file | FileCheck %s

// -----

acc.private.recipe @priv_ref_box : !fir.box<i32> init {
^bb0(%arg0: !fir.box<i32>):
  %1 = fir.allocmem i32
  %2 = fir.embox %1 : (!fir.heap<i32>) -> !fir.box<i32>
  acc.yield %2 : !fir.box<i32>
} destroy {
^bb0(%arg0: !fir.box<i32>, %arg1: !fir.box<i32>):
  %0 = fir.box_addr %arg1 : (!fir.box<i32>) -> !fir.ref<i32>
  %1 = fir.convert %0 : (!fir.ref<i32>) -> !fir.heap<i32>
  fir.freemem %1 : !fir.heap<i32>
  acc.yield
}

// CHECK-LABEL: acc.private.recipe @priv_ref_box : !fir.ref<!fir.box<i32>> init
// CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOX:.*]] = fir.embox
// CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK: } destroy {
// CHECK: ^bb0(%[[DARG0:.*]]: !fir.ref<!fir.box<i32>>, %[[DARG1:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[LD1:.*]] = fir.load %[[DARG1]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[ADDR:.*]] = fir.box_addr %[[LD1]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[CVT:.*]] = fir.convert %[[ADDR]] : (!fir.ref<i32>) -> !fir.heap<i32>

// -----

// Test private recipe without destroy region.

acc.private.recipe @priv_ref_box_no_destroy : !fir.box<i32> init {
^bb0(%arg0: !fir.box<i32>):
  %1 = fir.alloca i32
  %2 = fir.embox %1 : (!fir.ref<i32>) -> !fir.box<i32>
  acc.yield %2 : !fir.box<i32>
}

// CHECK-LABEL: acc.private.recipe @priv_ref_box_no_destroy : !fir.ref<!fir.box<i32>> init
// CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOX:.*]] = fir.embox
// CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK: }

// -----

// Firstprivate recipe with destroy region.
acc.firstprivate.recipe @fp_ref_box : !fir.box<i32> init {
^bb0(%arg0: !fir.box<i32>):
  %0 = fir.allocmem i32
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<i32>
  acc.yield %1 : !fir.box<i32>
} copy {
^bb0(%src: !fir.box<i32>, %dst: !fir.box<i32>):
  %s_addr = fir.box_addr %src : (!fir.box<i32>) -> !fir.ref<i32>
  %val = fir.load %s_addr : !fir.ref<i32>
  %d_addr = fir.box_addr %dst : (!fir.box<i32>) -> !fir.ref<i32>
  fir.store %val to %d_addr : !fir.ref<i32>
  acc.yield
} destroy {
^bb0(%arg0: !fir.box<i32>, %arg1: !fir.box<i32>):
  acc.yield
}

// CHECK-LABEL: acc.firstprivate.recipe @fp_ref_box : !fir.ref<!fir.box<i32>> init
// CHECK: ^bb0(%[[IARG:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOX_FP:.*]] = fir.embox
// CHECK:   %[[ALLOCA_FP:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOX_FP]] to %[[ALLOCA_FP]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCA_FP]] : !fir.ref<!fir.box<i32>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.box<i32>>, %[[DST:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[LSRC:.*]] = fir.load %[[SRC]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LDST:.*]] = fir.load %[[DST]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[SADDR:.*]] = fir.box_addr %[[LSRC]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[VAL:.*]] = fir.load %[[SADDR]] : !fir.ref<i32>
// CHECK:   %[[DADDR:.*]] = fir.box_addr %[[LDST]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   fir.store %[[VAL]] to %[[DADDR]] : !fir.ref<i32>
// CHECK: } destroy {
// CHECK: ^bb0(%[[FDARG0:.*]]: !fir.ref<!fir.box<i32>>, %[[FDARG1:.*]]: !fir.ref<!fir.box<i32>>)

// -----

// Firstprivate recipe without destroy region.
acc.firstprivate.recipe @fp_ref_box_no_destroy : !fir.box<i32> init {
^bb0(%arg0: !fir.box<i32>):
  %0 = fir.alloca i32
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  acc.yield %1 : !fir.box<i32>
} copy {
^bb0(%src: !fir.box<i32>, %dst: !fir.box<i32>):
  %s_addr = fir.box_addr %src : (!fir.box<i32>) -> !fir.ref<i32>
  %val = fir.load %s_addr : !fir.ref<i32>
  %d_addr = fir.box_addr %dst : (!fir.box<i32>) -> !fir.ref<i32>
  fir.store %val to %d_addr : !fir.ref<i32>
  acc.yield
}

// CHECK-LABEL: acc.firstprivate.recipe @fp_ref_box_no_destroy : !fir.ref<!fir.box<i32>> init
// CHECK: ^bb0(%[[IARG2:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOX_FP2:.*]] = fir.embox
// CHECK:   %[[ALLOCA_FP2:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOX_FP2]] to %[[ALLOCA_FP2]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCA_FP2]] : !fir.ref<!fir.box<i32>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC2:.*]]: !fir.ref<!fir.box<i32>>, %[[DST2:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[LSRC2:.*]] = fir.load %[[SRC2]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LDST2:.*]] = fir.load %[[DST2]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[SADDR2:.*]] = fir.box_addr %[[LSRC2]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[VAL2:.*]] = fir.load %[[SADDR2]] : !fir.ref<i32>
// CHECK:   %[[DADDR2:.*]] = fir.box_addr %[[LDST2]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   fir.store %[[VAL2]] to %[[DADDR2]] : !fir.ref<i32>

// -----

// Reduction recipe with destroy region.
acc.reduction.recipe @red_ref_box : !fir.box<i32> reduction_operator <add> init {
^bb0(%arg0: !fir.box<i32>):
  %0 = fir.allocmem i32
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<i32>
  acc.yield %1 : !fir.box<i32>
} combiner {
^bb0(%lhs: !fir.box<i32>, %rhs: !fir.box<i32>):
  %l_addr = fir.box_addr %lhs : (!fir.box<i32>) -> !fir.ref<i32>
  %l_val = fir.load %l_addr : !fir.ref<i32>
  %r_addr = fir.box_addr %rhs : (!fir.box<i32>) -> !fir.ref<i32>
  %r_val = fir.load %r_addr : !fir.ref<i32>
  %sum = arith.addi %l_val, %r_val : i32
  %tmp = fir.alloca i32
  fir.store %sum to %tmp : !fir.ref<i32>
  %new = fir.embox %tmp : (!fir.ref<i32>) -> !fir.box<i32>
  acc.yield %new : !fir.box<i32>
} destroy {
^bb0(%arg0: !fir.box<i32>, %arg1: !fir.box<i32>):
  acc.yield
}

// CHECK-LABEL: acc.reduction.recipe @red_ref_box : !fir.ref<!fir.box<i32>> reduction_operator <add> init
// CHECK: ^bb0(%[[IARGR:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOXR:.*]] = fir.embox
// CHECK:   %[[ALLOCAR:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOXR]] to %[[ALLOCAR]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCAR]] : !fir.ref<!fir.box<i32>>
// CHECK: } combiner {
// CHECK: ^bb0(%[[LHS:.*]]: !fir.ref<!fir.box<i32>>, %[[RHS:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[LLHS:.*]] = fir.load %[[LHS]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LRHS:.*]] = fir.load %[[RHS]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LADDR:.*]] = fir.box_addr %[[LLHS]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[LVAL:.*]] = fir.load %[[LADDR]] : !fir.ref<i32>
// CHECK:   %[[RADDR:.*]] = fir.box_addr %[[LRHS]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[RVAL:.*]] = fir.load %[[RADDR]] : !fir.ref<i32>
// CHECK:   %[[SUM:.*]] = arith.addi %[[LVAL]], %[[RVAL]] : i32
// CHECK:   %[[I32ALLOCA:.*]] = fir.alloca i32
// CHECK:   fir.store %[[SUM]] to %[[I32ALLOCA]] : !fir.ref<i32>
// CHECK:   %[[NEWBOX:.*]] = fir.embox %[[I32ALLOCA]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:   %[[BOXALLOCA:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[NEWBOX]] to %[[BOXALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[BOXALLOCA]] : !fir.ref<!fir.box<i32>>
// CHECK: } destroy {
// CHECK: ^bb0(%[[RD0:.*]]: !fir.ref<!fir.box<i32>>, %[[RD1:.*]]: !fir.ref<!fir.box<i32>>)

// -----

// Reduction recipe without destroy region.
acc.reduction.recipe @red_ref_box_no_destroy : !fir.box<i32> reduction_operator <add> init {
^bb0(%arg0: !fir.box<i32>):
  %0 = fir.alloca i32
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  acc.yield %1 : !fir.box<i32>
} combiner {
^bb0(%lhs: !fir.box<i32>, %rhs: !fir.box<i32>):
  %l_addr = fir.box_addr %lhs : (!fir.box<i32>) -> !fir.ref<i32>
  %l_val = fir.load %l_addr : !fir.ref<i32>
  %r_addr = fir.box_addr %rhs : (!fir.box<i32>) -> !fir.ref<i32>
  %r_val = fir.load %r_addr : !fir.ref<i32>
  %sum = arith.addi %l_val, %r_val : i32
  %tmp = fir.alloca i32
  fir.store %sum to %tmp : !fir.ref<i32>
  %new = fir.embox %tmp : (!fir.ref<i32>) -> !fir.box<i32>
  acc.yield %new : !fir.box<i32>
}

// CHECK-LABEL: acc.reduction.recipe @red_ref_box_no_destroy : !fir.ref<!fir.box<i32>> reduction_operator <add> init
// CHECK: ^bb0(%[[IARGR2:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[EMBOXR2:.*]] = fir.embox
// CHECK:   %[[ALLOCAR2:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[EMBOXR2]] to %[[ALLOCAR2]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[ALLOCAR2]] : !fir.ref<!fir.box<i32>>
// CHECK: } combiner {
// CHECK: ^bb0(%[[LHS2:.*]]: !fir.ref<!fir.box<i32>>, %[[RHS2:.*]]: !fir.ref<!fir.box<i32>>)
// CHECK:   %[[LLHS2:.*]] = fir.load %[[LHS2]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LRHS2:.*]] = fir.load %[[RHS2]] : !fir.ref<!fir.box<i32>>
// CHECK:   %[[LADDR2:.*]] = fir.box_addr %[[LLHS2]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[LVAL2:.*]] = fir.load %[[LADDR2]] : !fir.ref<i32>
// CHECK:   %[[RADDR2:.*]] = fir.box_addr %[[LRHS2]] : (!fir.box<i32>) -> !fir.ref<i32>
// CHECK:   %[[RVAL2:.*]] = fir.load %[[RADDR2]] : !fir.ref<i32>
// CHECK:   %[[SUM2:.*]] = arith.addi %[[LVAL2]], %[[RVAL2]] : i32
// CHECK:   %[[I32ALLOCA2:.*]] = fir.alloca i32
// CHECK:   fir.store %[[SUM2]] to %[[I32ALLOCA2]] : !fir.ref<i32>
// CHECK:   %[[NEWBOX2:.*]] = fir.embox %[[I32ALLOCA2]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:   %[[BOXALLOCA2:.*]] = fir.alloca !fir.box<i32>
// CHECK:   fir.store %[[NEWBOX2]] to %[[BOXALLOCA2]] : !fir.ref<!fir.box<i32>>
// CHECK:   acc.yield %[[BOXALLOCA2]] : !fir.ref<!fir.box<i32>>

// -----

// Comprehensive tests that also test recipe usages updates.

acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
^bb0(%arg0: !fir.ref<i32>):
  %0 = fir.alloca i32
  %1 = fir.declare %0 {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> !fir.ref<i32>
  acc.yield %1 : !fir.ref<i32>
}
acc.private.recipe @privatization_box_Uxf32 : !fir.box<!fir.array<?xf32>> init {
^bb0(%arg0: !fir.box<!fir.array<?xf32>>):
  %c0 = arith.constant 0 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %1 = fir.shape %0#1 : (index) -> !fir.shape<1>
  %2 = fir.allocmem !fir.array<?xf32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
  %3 = fir.declare %2(%1) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xf32>>
  %4 = fir.embox %3(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  acc.yield %4 : !fir.box<!fir.array<?xf32>>
} destroy {
^bb0(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>):
  %0 = fir.box_addr %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %1 = fir.convert %0 : (!fir.ref<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
  fir.freemem %1 : !fir.heap<!fir.array<?xf32>>
  acc.terminator
}
func.func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c200_i32 = arith.constant 200 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFfooEi"}
  %2 = fir.declare %1 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  acc.parallel combined(loop) {
    %4 = acc.private var(%3 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {name = "x"}
    %5 = acc.private varPtr(%2 : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
    acc.loop combined(parallel) private(@privatization_box_Uxf32 -> %4 : !fir.box<!fir.array<?xf32>>, @privatization_ref_i32 -> %5 : !fir.ref<i32>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c200_i32 : i32)  step (%c1_i32 : i32) {
      %6 = fir.dummy_scope : !fir.dscope
      %7 = fir.declare %4 dummy_scope %6 {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
      %8 = fir.declare %5 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %9 = fir.convert %arg1 : (i32) -> f32
      %10 = fir.convert %arg1 : (i32) -> i64
      %11 = fir.array_coor %7 %10 : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      fir.store %9 to %11 : !fir.ref<f32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL:   acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
// CHECK:           %[[VAL_1:.*]] = fir.alloca i32
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           acc.yield %[[VAL_2]] : !fir.ref<i32>
// CHECK:         }

// CHECK-LABEL:   acc.private.recipe @privatization_box_Uxf32 : !fir.ref<!fir.box<!fir.array<?xf32>>> init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>):
// CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_3]]#1 {bindc_name = ".tmp", uniq_name = ""}
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]](%[[VAL_4]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]](%[[VAL_4]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_7]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:           acc.yield %[[VAL_8]] : !fir.ref<!fir.box<!fir.array<?xf32>>>

// CHECK-LABEL:   } destroy {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>):
// CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           fir.freemem %[[VAL_4]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           acc.terminator
// CHECK:         }

// CHECK-LABEL:   func.func @_QPfoo(
// CHECK-SAME:                      %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 200 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFfooEi"}
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_3]] {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:           acc.parallel combined(loop) {
// CHECK:             %[[VAL_7:.*]] = acc.private varPtr(%[[VAL_6]] : !fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<!fir.array<?xf32>>> {name = "x"}
// CHECK:             %[[VAL_8:.*]] = acc.private varPtr(%[[VAL_4]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
// CHECK:             acc.loop combined(parallel) private(@privatization_box_Uxf32 -> %[[VAL_7]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, @privatization_ref_i32 -> %[[VAL_8]] : !fir.ref<i32>) control(%[[VAL_9:.*]] : i32) = (%[[VAL_1]] : i32) to (%[[VAL_0]] : i32)  step (%[[VAL_1]] : i32) {
// CHECK:               %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:               %[[VAL_11:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:               %[[VAL_12:.*]] = fir.declare %[[VAL_11]] dummy_scope %[[VAL_10]] {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:               %[[VAL_13:.*]] = fir.declare %[[VAL_8]] {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (i32) -> f32
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:               %[[VAL_16:.*]] = fir.array_coor %[[VAL_12]] %[[VAL_15]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK:               fir.store %[[VAL_14]] to %[[VAL_16]] : !fir.ref<f32>
// CHECK:               acc.yield
// CHECK:             } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
