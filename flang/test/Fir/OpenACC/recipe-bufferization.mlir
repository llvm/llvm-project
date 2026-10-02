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
    %4 = acc.private var(%3 : !fir.box<!fir.array<?xf32>>) recipe(@privatization_box_Uxf32) name("x") -> !fir.box<!fir.array<?xf32>>
    %5 = acc.private varPtr(%2 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i") -> !fir.ref<i32>
    acc.loop combined(parallel) private(%4, %5 : !fir.box<!fir.array<?xf32>>, !fir.ref<i32>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c200_i32 : i32)  step (%c1_i32 : i32) {
      %6 = fir.dummy_scope : !fir.dscope
      %7 = fir.declare %4 dummy_scope %6 {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
      %8 = fir.declare %5 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %9 = fir.convert %arg1 : (i32) -> f32
      %10 = fir.convert %arg1 : (i32) -> i64
      %11 = fir.array_coor %7 %10 : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      fir.store %9 to %11 : !fir.ref<f32>
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
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
// CHECK:             %[[VAL_7:.*]] = acc.private varPtr(%[[VAL_6]] : !fir.ref<!fir.box<!fir.array<?xf32>>>) recipe(@privatization_box_Uxf32) name("x") -> !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:             %[[VAL_8:.*]] = acc.private varPtr(%[[VAL_4]] : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i") -> !fir.ref<i32>
// CHECK:             acc.loop combined(parallel) private(%[[VAL_7]], %[[VAL_8]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.ref<i32>) control(%[[VAL_9:.*]] : i32) = (%[[VAL_1]] : i32) to (%[[VAL_0]] : i32)  step (%[[VAL_1]] : i32) {
// CHECK:               %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:               %[[VAL_11:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
// CHECK:               %[[VAL_12:.*]] = fir.declare %[[VAL_11]] dummy_scope %[[VAL_10]] {uniq_name = "_QFfooEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:               %[[VAL_13:.*]] = fir.declare %[[VAL_8]] {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (i32) -> f32
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:               %[[VAL_16:.*]] = fir.array_coor %[[VAL_12]] %[[VAL_15]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK:               fir.store %[[VAL_14]] to %[[VAL_16]] : !fir.ref<f32>
// CHECK:               acc.yield
// CHECK:             } inclusiveUpperbound(array<i1: true>) independent
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

// A loop reduction on a mapped descriptor. The memory holding the descriptor
// is created in the construct region, where it is device memory read through
// the mapped descriptor and needs no data clause of its own.

acc.reduction.recipe @red_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: !fir.box<!fir.array<?xi32>>):
  acc.yield %lhs : !fir.box<!fir.array<?xi32>>
}
func.func @_QPloop_reduction(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFloop_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) {
    %2 = acc.reduction var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@red_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang reduction(%2 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPloop_reduction(
// CHECK-SAME:                                 %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
// CHECK:           %[[LB:.*]] = arith.constant 1 : i32
// CHECK:           %[[UB:.*]] = arith.constant 32 : i32
// CHECK:           %[[DECL:.*]] = fir.declare %[[ARG0]] {uniq_name = "_QFloop_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%[[DECL]] : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        %[[RED:.*]] = acc.reduction varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@red_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             acc.loop gang reduction(%[[RED]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(%{{.*}} : i32) = (%[[LB]] : i32) to (%[[UB]] : i32)  step (%[[LB]] : i32) {
// CHECK:             } inclusiveUpperbound(array<i1: true>) independent
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

// A reduction carried by a nested loop. The memory belongs at the top of the
// construct region rather than next to the clause, so that it is not
// allocated on every iteration of the enclosing loop.

acc.reduction.recipe @red_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: !fir.box<!fir.array<?xi32>>):
  acc.yield %lhs : !fir.box<!fir.array<?xi32>>
}
func.func @_QPnested_loop_reduction(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFnested_loop_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) {
    acc.loop gang control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      %2 = acc.reduction var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@red_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
      acc.loop vector reduction(%2 : !fir.box<!fir.array<?xi32>>) control(%arg2 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
        acc.yield
      } inclusiveUpperbound(array<i1: true>) independent
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPnested_loop_reduction(
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        acc.loop gang control(
// CHECK:               %[[RED:.*]] = acc.reduction varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@red_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:               acc.loop vector reduction(%[[RED]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(
// CHECK:           return
// CHECK:         }

// -----

// A reduction on the compute construct itself has to keep its memory outside
// the construct: the clause is an operand of the construct and so cannot refer
// to a value defined in its region.

acc.reduction.recipe @red_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: !fir.box<!fir.array<?xi32>>):
  acc.yield %lhs : !fir.box<!fir.array<?xi32>>
}
func.func @_QPconstruct_reduction(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %0 = fir.declare %arg0 {uniq_name = "_QFconstruct_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  %2 = acc.reduction var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@red_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) reduction(%2 : !fir.box<!fir.array<?xi32>>) {
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPconstruct_reduction(
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:           fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:           %[[RED:.*]] = acc.reduction varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@red_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) reduction(%[[RED]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) {

// -----

// A descriptor that is not mapped keeps its memory outside the construct: it
// is not a live-in of the region because only private clauses use it, and
// storing it in the region would make it one.

acc.private.recipe @priv_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
}
func.func @_QPunmapped_private(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFunmapped_privateEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  acc.parallel {
    %1 = acc.private var(%0 : !fir.box<!fir.array<?xi32>>) recipe(@priv_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang private(%1 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPunmapped_private(
// CHECK:           %[[DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFunmapped_privateEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:           fir.store %[[DECL]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:           acc.parallel {
// CHECK-NEXT:        %[[PRIV:.*]] = acc.private varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@priv_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             acc.loop gang private(%[[PRIV]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(

// -----

// Placement follows the descriptor value, not the kind of clause: a private
// clause on a mapped descriptor gets its memory in the construct region for
// the same reason a reduction does.

acc.private.recipe @priv_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
}
func.func @_QPmapped_private(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFmapped_privateEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) {
    %2 = acc.private var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@priv_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang private(%2 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPmapped_private(
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        %[[PRIV:.*]] = acc.private varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@priv_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             acc.loop gang private(%[[PRIV]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(

// -----

// The same for firstprivate.

acc.firstprivate.recipe @fp_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} copy {
^bb0(%src: !fir.box<!fir.array<?xi32>>, %dst: !fir.box<!fir.array<?xi32>>):
  acc.terminator
}
func.func @_QPmapped_firstprivate(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFmapped_firstprivateEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) {
    %2 = acc.firstprivate var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@fp_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang firstprivate(%2 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPmapped_firstprivate(
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        %[[FP:.*]] = acc.firstprivate varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@fp_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             acc.loop gang firstprivate(%[[FP]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(

// -----

// A clause that refers to a mapped descriptor through a declare in the region
// keeps the default placement next to that declare. The value stored is the
// declare result, so the memory cannot be hoisted above it, and the declare
// is already inside the region.

acc.private.recipe @priv_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
}
func.func @_QPmapped_private_via_declare(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFmapped_private_via_declareEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xi32>>) {
    %2 = fir.declare %1 {uniq_name = "_QFmapped_private_via_declareEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %3 = acc.private var(%2 : !fir.box<!fir.array<?xi32>>) recipe(@priv_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang private(%3 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPmapped_private_via_declare(
// CHECK:           %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK:           acc.parallel dataOperands(%[[MAPPED]] : !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:        %[[INNER:.*]] = fir.declare %[[MAPPED]] {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[INNER]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        %[[PRIV:.*]] = acc.private varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@priv_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>

// -----

// A data entry operation inside the construct also keeps the default
// placement: hoisting the memory to the top of the region would place it
// above the value it stores.

acc.reduction.recipe @red_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: !fir.box<!fir.array<?xi32>>):
  acc.yield %lhs : !fir.box<!fir.array<?xi32>>
}
func.func @_QPin_region_data_entry(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFin_region_data_entryEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  acc.parallel {
    %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
    %2 = acc.reduction var(%1 : !fir.box<!fir.array<?xi32>>) recipe(@red_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang reduction(%2 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPin_region_data_entry(
// CHECK:           acc.parallel {
// CHECK-NEXT:        %[[MAPPED:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xi32>>) dataClause(acc_copy) implicit(true) name("r") -> !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:        fir.store %[[MAPPED]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK-NEXT:        %[[RED:.*]] = acc.reduction varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@red_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>

// -----

// A reduction on a descriptor that is not mapped keeps the default placement
// in host code, like an unmapped private clause.
//
// This shape does not arise from a compiler pipeline that applies implicit
// data clauses, because any aggregate live-in receives one, so a reduction on
// a box is already mapped when this pass runs. It would not work if it did
// arise: a reduction gets no initial value mapping, so nothing maps the
// memory wherever it is placed. The case pins the condition for the reduction
// clause rather than describing a working end state.

acc.reduction.recipe @red_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.array<?xi32>>):
  acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: !fir.box<!fir.array<?xi32>>):
  acc.yield %lhs : !fir.box<!fir.array<?xi32>>
}
func.func @_QPunmapped_reduction(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "r"}) {
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = fir.declare %arg0 {uniq_name = "_QFunmapped_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  acc.parallel {
    %1 = acc.reduction var(%0 : !fir.box<!fir.array<?xi32>>) recipe(@red_box_Uxi32) name("r") -> !fir.box<!fir.array<?xi32>>
    acc.loop gang reduction(%1 : !fir.box<!fir.array<?xi32>>) control(%arg1 : i32) = (%c1_i32 : i32) to (%c32_i32 : i32)  step (%c1_i32 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.yield
  }
  return
}

// CHECK-LABEL:   func.func @_QPunmapped_reduction(
// CHECK:           %[[DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFunmapped_reductionEr"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:           fir.store %[[DECL]] to %[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:           acc.parallel {
// CHECK-NEXT:        %[[RED:.*]] = acc.reduction varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) recipe(@red_box_Uxi32) name("r") -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             acc.loop gang reduction(%[[RED]] : !fir.ref<!fir.box<!fir.array<?xi32>>>) control(
