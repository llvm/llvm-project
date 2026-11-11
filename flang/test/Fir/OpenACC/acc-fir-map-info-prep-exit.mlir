// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// The map entry replaces both the data entry operation and the data exit
// operations paired with it: its flags hold the effects of both, and it keeps
// the location where the exit effects happen - the end directive of a
// structured construct.

// CHECK: #[[END_DATA:.*]] = loc("end-data":8:4)

// CHECK-LABEL: func.func @structured_copy
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<f32>)
// CHECK-SAME: exitLoc(#[[END_DATA]])
// CHECK-SAME: mapFlags(to,from)
// CHECK-NOT: acc.copyin
// CHECK-NOT: acc.copyout
func.func @structured_copy() {
  %ref = fir.undefined !fir.ref<f32>
  %copy = acc.copyin varPtr(%ref : !fir.ref<f32>) dataClause(acc_copy)
      name("x") -> !fir.ref<f32> loc("data":4:2)
  acc.data dataOperands(%copy : !fir.ref<f32>) {
    acc.terminator
  }
  acc.copyout accPtr(%copy : !fir.ref<f32>) to varPtr(%ref : !fir.ref<f32>)
      dataClause(acc_copy) name("x") loc("end-data":8:4)
  return
}

// A construct that maps one variable through two clauses, as `!$acc data
// copyin(x) copyout(x)` does. The runtime keeps a single mapping, so the
// copy-in entry - whose own exit only releases the device copy - has to carry
// the copy-back of its sibling.

// CHECK-LABEL: func.func @aliased_clauses
// CHECK: acc.map_info var(%[[BOX:.*]] : !fir.box<!fir.array<?xf32>>)
// CHECK-SAME: mapFlags(to,from)
// CHECK: acc.map_info var(%[[BOX]] : !fir.box<!fir.array<?xf32>>)
// CHECK-SAME: mapFlags(from)
// CHECK-NOT: acc.delete
// CHECK-NOT: acc.copyout
func.func @aliased_clauses(%box: !fir.box<!fir.array<?xf32>>) {
  %copyin = acc.copyin var(%box : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_copyin) name("x") -> !fir.box<!fir.array<?xf32>>
  %create = acc.create var(%box : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_copyout) name("x") -> !fir.box<!fir.array<?xf32>>
  acc.data dataOperands(
      %copyin, %create : !fir.box<!fir.array<?xf32>>,
      !fir.box<!fir.array<?xf32>>) {
    acc.terminator
  }
  acc.delete accVar(%copyin : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_copyin) name("x")
  acc.copyout accVar(%create : !fir.box<!fir.array<?xf32>>)
      to var(%box : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_copyout) name("x")
  return
}

// A readonly copy-in is released, not copied back: its exit repeats the entry
// clause and no sibling clause on the construct copies the variable out, so the
// mapping stays copy-to-device only.

// CHECK-LABEL: func.func @copyin_readonly_release
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<f32>)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: acc.delete
func.func @copyin_readonly_release() {
  %ref = fir.undefined !fir.ref<f32>
  %copyin = acc.copyin varPtr(%ref : !fir.ref<f32>)
      dataClause(acc_copyin_readonly) name("x") -> !fir.ref<f32>
  acc.data dataOperands(%copyin : !fir.ref<f32>) {
    acc.terminator
  }
  acc.delete accPtr(%copyin : !fir.ref<f32>) dataClause(acc_copyin_readonly)
      name("x")
  return
}

// -----

// `!$acc declare copy(a)` uses one data entry operation at two program points:
// acc.declare_enter and acc.declare_exit. Both keep referring to the single map
// entry, which holds the union of the entry and exit effects - the same shape a
// structured acc.data region has. Each declare call site uses the same map_info
// token with the flags that apply there, rather than splitting the entry.

// CHECK-LABEL: func.func @declare_copy
// CHECK: %[[MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<100xi32>>)
// CHECK-SAME: mapFlags(to,from)
// CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK-NOT: acc.copyout
func.func @declare_copy(%decl: !fir.ref<!fir.array<100xi32>>) {
  %copyin = acc.copyin varPtr(%decl : !fir.ref<!fir.array<100xi32>>)
      dataClause(acc_copy) name("a") -> !fir.ref<!fir.array<100xi32>>
  %token = acc.declare_enter dataOperands(%copyin : !fir.ref<!fir.array<100xi32>>)
  acc.declare_exit token(%token) dataOperands(%copyin : !fir.ref<!fir.array<100xi32>>)
  acc.copyout accPtr(%copyin : !fir.ref<!fir.array<100xi32>>)
      to varPtr(%decl : !fir.ref<!fir.array<100xi32>>) dataClause(acc_copy)
      name("a")
  return
}

// -----

// Only device_resident needs the entry and the exit to differ: the exit adds
// the teardown of the mapping. The map entry records device_resident once and
// the exit call site adds that teardown, so the entry stays unsplit here too.

// CHECK-LABEL: func.func @declare_device_resident
// CHECK: %[[MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<100xi32>>)
// CHECK-SAME: mapFlags(device_resident)
// CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK-NOT: acc.delete
func.func @declare_device_resident(%decl: !fir.ref<!fir.array<100xi32>>) {
  %dr = acc.declare_device_resident varPtr(%decl : !fir.ref<!fir.array<100xi32>>)
      name("a") -> !fir.ref<!fir.array<100xi32>>
  %token = acc.declare_enter dataOperands(%dr : !fir.ref<!fir.array<100xi32>>)
  acc.declare_exit token(%token) dataOperands(%dr : !fir.ref<!fir.array<100xi32>>)
  acc.delete accPtr(%dr : !fir.ref<!fir.array<100xi32>>)
      dataClause(acc_declare_device_resident) name("a")
  return
}

// A declare map and a kernel use of the same variable need different map flags:
// declare_enter keeps device_resident; the kernel must use present instead.
// CHECK-LABEL: func.func @declare_device_resident_kernel_present
// CHECK: %[[DECLARE_MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<100xi32>>)
// CHECK-SAME: mapFlags(device_resident)
// CHECK: %[[KERNEL_MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<100xi32>>)
// CHECK-SAME: mapFlags(present)
// CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[DECLARE_MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK: acc.kernel_environment dataOperands(%[[KERNEL_MAP]] : !fir.ref<!fir.array<100xi32>>)
// CHECK-NOT: acc.declare_device_resident
func.func @declare_device_resident_kernel_present(%decl: !fir.ref<!fir.array<100xi32>>) {
  %dr = acc.declare_device_resident varPtr(%decl : !fir.ref<!fir.array<100xi32>>)
      name("a") -> !fir.ref<!fir.array<100xi32>>
  %token = acc.declare_enter dataOperands(%dr : !fir.ref<!fir.array<100xi32>>)
  acc.kernel_environment dataOperands(%dr : !fir.ref<!fir.array<100xi32>>) {
  }
  return
}

// -----

// An unstructured exit_data copy-out with finalize becomes one map entry whose
// flags hold both the copy-back and the delete, and the exit_data construct
// keeps that token as its operand.

// CHECK-LABEL: func.func @unstructured_exit_data
// CHECK: %[[MAP:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<f32>)
// CHECK-SAME: mapFlags(from,delete)
// CHECK: acc.exit_data dataOperands(%[[MAP]] : !fir.ref<f32>)
// CHECK-NOT: acc.copyout
func.func @unstructured_exit_data() {
  %ref = fir.undefined !fir.ref<f32>
  %devptr = acc.getdeviceptr varPtr(%ref : !fir.ref<f32>)
      dataClause(acc_copyout) structured(false) name("x") -> !fir.ref<f32>
  acc.exit_data dataOperands(%devptr : !fir.ref<f32>) finalize
  acc.copyout accPtr(%devptr : !fir.ref<f32>) to varPtr(%ref : !fir.ref<f32>)
      dataClause(acc_copyout) structured(false) name("x")
  return
}

// -----

// `delete` decrements the dynamic reference counter of the mapping, so it must
// not ask the runtime to force an unmap: an enclosing data region may still
// hold a reference to the same memory, and tearing the mapping down here makes
// a later `present` on that memory fail. Only `finalize` zeroes the counter.

// CHECK-LABEL: func.func @unstructured_delete
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
// CHECK-SAME: mapFlags(ptr_and_obj)
// CHECK: acc.exit_data dataOperands
// CHECK-NOT: acc.delete
func.func @unstructured_delete(%box: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>) {
  %devptr = acc.getdeviceptr varPtr(%box : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
      dataClause(acc_delete) structured(false) name("dwork")
      -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>
  acc.exit_data dataOperands(%devptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
  acc.delete accPtr(%devptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
      structured(false) name("dwork")
  return
}

// -----

// The same `delete` clause under `finalize` does force the unmap.

// CHECK-LABEL: func.func @unstructured_delete_finalize
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
// CHECK-SAME: mapFlags(delete,ptr_and_obj)
// CHECK: acc.exit_data dataOperands(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>) finalize
// CHECK-NOT: acc.delete
func.func @unstructured_delete_finalize(%box: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>) {
  %devptr = acc.getdeviceptr varPtr(%box : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
      dataClause(acc_delete) structured(false) name("dwork")
      -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>
  acc.exit_data dataOperands(%devptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>) finalize
  acc.delete accPtr(%devptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf64>>>>)
      structured(false) name("dwork")
  return
}

// -----

// Reduction mapping records both transfer directions and the reduction
// behavior on the map entry.
// CHECK-LABEL: func.func @reduction
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<i32>)
// CHECK-SAME: mapFlags(to,from,reduction)
// CHECK-NOT: acc.copyout
func.func @reduction(%ref: !fir.ref<i32>) {
  %map = acc.copyin varPtr(%ref : !fir.ref<i32>)
      dataClause(acc_reduction) name("sum") -> !fir.ref<i32>
  acc.kernel_environment dataOperands(%map : !fir.ref<i32>) {
  }
  acc.copyout accPtr(%map : !fir.ref<i32>) to varPtr(%ref : !fir.ref<i32>)
      dataClause(acc_reduction) name("sum")
  return
}
