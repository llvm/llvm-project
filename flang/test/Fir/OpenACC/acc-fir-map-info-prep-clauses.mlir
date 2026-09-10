// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// Exercise entry operations that are not ordinary structured copy clauses.

// CHECK-LABEL: func.func @update_if_present
// CHECK: %[[TO:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>)
// CHECK-SAME: bounds(
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(to,if_present)
// CHECK: acc.update dataOperands(%[[TO]]
// CHECK: %[[FROM:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>)
// CHECK-SAME: bounds(
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(from,if_present)
// CHECK: acc.update dataOperands(%[[FROM]]
// CHECK-NOT: acc.update_device
// CHECK-NOT: acc.update_host
func.func @update_if_present() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %array = fir.undefined !fir.ref<!fir.array<10xi32>>
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
      extent(%c10 : index) stride(%c1 : index) startIdx(%c1 : index)
  %to = acc.update_device varPtr(%array : !fir.ref<!fir.array<10xi32>>)
      bounds(%bounds) structured(false) name("a")
      -> !fir.ref<!fir.array<10xi32>>
  acc.update dataOperands(%to : !fir.ref<!fir.array<10xi32>>) ifPresent
  %from = acc.getdeviceptr varPtr(%array : !fir.ref<!fir.array<10xi32>>)
      bounds(%bounds) dataClause(acc_update_host) structured(false) name("a")
      -> !fir.ref<!fir.array<10xi32>>
  acc.update dataOperands(%from : !fir.ref<!fir.array<10xi32>>) ifPresent
  acc.update_host accPtr(%from : !fir.ref<!fir.array<10xi32>>)
      bounds(%bounds) to varPtr(%array : !fir.ref<!fir.array<10xi32>>)
      structured(false) name("a")
  return
}

// CHECK-LABEL: func.func @nocreate
// CHECK: %[[MAP:.*]] = acc.map_info var(%{{.*}} : !fir.box<!fir.array<?xf32>>)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(no_create)
// CHECK: acc.data dataOperands(%[[MAP]]
// CHECK-NOT: acc.nocreate
// CHECK-NOT: acc.delete
func.func @nocreate() {
  %box = fir.undefined !fir.box<!fir.array<?xf32>>
  %map = acc.nocreate var(%box : !fir.box<!fir.array<?xf32>>)
      name("a") -> !fir.box<!fir.array<?xf32>>
  acc.data dataOperands(%map : !fir.box<!fir.array<?xf32>>) {
    acc.terminator
  }
  acc.delete accVar(%map : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_no_create) name("a")
  return
}

// An unstructured declaration entry is rewritten like any other data operand,
// while the declaration directive remains attached to the map result.
// CHECK-LABEL: func.func @declare_create
// CHECK: %[[SIZE:.*]] = arith.constant 28 : i64
// CHECK: %[[MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<7xf32>>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(none)
// CHECK: acc.declare_enter dataOperands(%[[MAP]]
// CHECK-NOT: acc.create
func.func @declare_create() {
  %array = fir.undefined !fir.ref<!fir.array<7xf32>>
  %map = acc.create varPtr(%array : !fir.ref<!fir.array<7xf32>>)
      structured(false) name("a") -> !fir.ref<!fir.array<7xf32>>
  %token = acc.declare_enter dataOperands(%map : !fir.ref<!fir.array<7xf32>>)
  acc.declare_exit token(%token) dataOperands(%map : !fir.ref<!fir.array<7xf32>>)
  return
}

// host_data requires acc.use_device to survive this pass; replacing it with a
// map_info would change the operation's device-address lookup semantics.
// CHECK-LABEL: func.func @keep_use_device
// CHECK: %[[USE:.*]] = acc.use_device
// CHECK-NOT: acc.map_info
// CHECK: acc.host_data dataOperands(%[[USE]]
func.func @keep_use_device() {
  %array = fir.undefined !fir.ref<!fir.array<4xf32>>
  %use = acc.use_device varPtr(%array : !fir.ref<!fir.array<4xf32>>)
      name("a") -> !fir.ref<!fir.array<4xf32>>
  acc.host_data dataOperands(%use : !fir.ref<!fir.array<4xf32>>) {
    acc.terminator
  }
  return
}

// An explicit map with no statically recoverable layout must remain unknown.
// Only an implicit present lookup is allowed to turn an unknown size into zero.
// CHECK-LABEL: func.func @explicit_unknown_size
// CHECK: %[[UNKNOWN:.*]] = arith.constant -1 : i64
// CHECK: acc.map_info
// CHECK-SAME: size(%[[UNKNOWN]] : i64)
// CHECK-SAME: mapFlags(to)
func.func @explicit_unknown_size() {
  %record = fir.undefined !fir.ref<!fir.type<_QFunknownTrecord{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  %copy = acc.copyin varPtr(%record : !fir.ref<!fir.type<_QFunknownTrecord{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
      dataClause(acc_copyin) name("record")
      -> !fir.ref<!fir.type<_QFunknownTrecord{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  acc.data dataOperands(%copy : !fir.ref<!fir.type<_QFunknownTrecord{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
    acc.terminator
  }
  return
}

// A device address supplied by the user is mapped as such: the runtime must not
// look it up or transfer it.
// CHECK-LABEL: func.func @deviceptr
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<4xf32>>)
// CHECK-SAME: mapFlags(devptr)
// CHECK-NOT: acc.deviceptr
func.func @deviceptr() {
  %array = fir.undefined !fir.ref<!fir.array<4xf32>>
  %map = acc.deviceptr varPtr(%array : !fir.ref<!fir.array<4xf32>>)
      name("a") -> !fir.ref<!fir.array<4xf32>>
  acc.data dataOperands(%map : !fir.ref<!fir.array<4xf32>>) {
    acc.terminator
  }
  return
}

// Detaching a pointer component acts on the descriptor and the address it
// holds, so the map keeps ptr_and_obj without requesting any transfer.
// CHECK-LABEL: func.func @detach
// CHECK: %[[MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(ptr_and_obj)
// CHECK: acc.exit_data dataOperands(%[[MAP]]
// CHECK-NOT: acc.detach
func.func @detach() {
  %box = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
  %ptr = acc.getdeviceptr varPtr(%box : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      dataClause(acc_detach) structured(false) name("p")
      -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  acc.exit_data dataOperands(%ptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
  acc.detach accPtr(%ptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      structured(false) name("p")
  return
}

// -----

// Storage that a recipe creates on the device is never a host mapping: the
// clause operations that carry those recipes stay as they are.
// CHECK-LABEL: func.func @recipe_clauses_stay
// CHECK: %[[PRIV:.*]] = acc.private
// CHECK: %[[RED:.*]] = acc.reduction
// CHECK: acc.parallel private(%[[PRIV]]
// CHECK-SAME: reduction(%[[RED]]
// CHECK-NOT: acc.map_info
acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
^bb0(%arg0: !fir.ref<i32>):
  %0 = fir.alloca i32
  acc.yield %0 : !fir.ref<i32>
}
acc.reduction.recipe @reduction_add_ref_i32 : !fir.ref<i32>
    reduction_operator <add> init {
^bb0(%arg0: !fir.ref<i32>):
  %0 = fir.alloca i32
  acc.yield %0 : !fir.ref<i32>
} combiner {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
  acc.yield %arg0 : !fir.ref<i32>
}
func.func @recipe_clauses_stay(%a: !fir.ref<i32>, %b: !fir.ref<i32>) {
  %priv = acc.private varPtr(%a : !fir.ref<i32>)
      recipe(@privatization_ref_i32) -> !fir.ref<i32>
  %red = acc.reduction varPtr(%b : !fir.ref<i32>)
      recipe(@reduction_add_ref_i32) -> !fir.ref<i32>
  acc.parallel private(%priv : !fir.ref<i32>)
      reduction(%red : !fir.ref<i32>) {
    acc.yield
  }
  return
}

// -----

// A cache hint describes storage the compiler may promote inside the loop, not
// a mapping the runtime performs.
// CHECK-LABEL: func.func @cache_stays
// CHECK: %[[CACHE:.*]] = acc.cache
// CHECK-NOT: acc.map_info
// CHECK: acc.loop {{.*}}cache(%[[CACHE]]
func.func @cache_stays(%a: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cache = acc.cache varPtr(%a : !fir.ref<!fir.array<10xf32>>)
      name("a") -> !fir.ref<!fir.array<10xf32>>
  acc.loop cache(%cache : !fir.ref<!fir.array<10xf32>>)
      control(%iv : index) = (%c0 : index) to (%c10 : index)
      step (%c1 : index) {
    acc.yield
  } inclusiveUpperbound(array<i1: true>) independent
  return
}

// -----

// CUDA Fortran managed allocatables set managed_devptr from cuf.data_attr.
// CHECK-LABEL: func.func @managed_copy
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>)
// CHECK-SAME: mapFlags(to,from,managed_devptr)
func.func @managed_copy() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32> {cuf.data_attr = #cuf.cuda<managed>}
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%shape) {cuf.data_attr = #cuf.cuda<managed>, uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
      extent(%c10 : index) stride(%c1 : index) startIdx(%c1 : index)
  %copy = acc.copyin varPtr(%1 : !fir.ref<!fir.array<10xf32>>) bounds(%bounds)
      dataClause(acc_copy) name("a") -> !fir.ref<!fir.array<10xf32>>
  acc.data dataOperands(%copy : !fir.ref<!fir.array<10xf32>>) {
    acc.terminator
  }
  acc.copyout accPtr(%copy : !fir.ref<!fir.array<10xf32>>) bounds(%bounds)
      to varPtr(%1 : !fir.ref<!fir.array<10xf32>>) dataClause(acc_copy) name("a")
  return
}
