// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// Implicit `deviceptr` classification keys on device residency, not device
// accessibility.
//
// A `managed`/`unified` attribute makes storage reachable from the device
// (device-accessible), but the storage is physically shared with the host and may
// migrate on demand, so it is not guaranteed to be in device memory. Such data 
// must be mapped so the runtime can attach rather than be  treated as `deviceptr`.
//
// `device` (and `constant`/`shared`) storage is physically in device memory and
// is therefore given `deviceptr`.

// A managed POINTER descriptor must be mapped, not deviceptr.
func.func @managed_ptr_desc() {
  %0 = cuf.alloc !fir.box<!fir.ptr<!fir.array<?xf64>>> {bindc_name = "p", data_attr = #cuf.cuda<managed>, uniq_name = "_QFEp"} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_ptr_desc
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)
// CHECK: acc.copyout accPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) to varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)

// -----

// A managed ALLOCATABLE descriptor must be mapped, not deviceptr (same rule).
func.func @managed_alloc_desc() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "a", data_attr = #cuf.cuda<managed>, uniq_name = "_QFEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_alloc_desc
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)
// CHECK: acc.copyout accPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) to varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)

// -----

// A unified ALLOCATABLE descriptor must be mapped, not deviceptr.
func.func @unified_alloc_desc() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "u", data_attr = #cuf.cuda<unified>, uniq_name = "_QFEu"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @unified_alloc_desc
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)

// -----

// A device descriptor is physically in device memory and stays deviceptr.
func.func @device_alloc_desc() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "d", data_attr = #cuf.cuda<device>, uniq_name = "_QFEd"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @device_alloc_desc
// CHECK-NOT: acc.copyin
// CHECK: acc.deviceptr varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) implicit(true)

// -----

// The underlying-storage walk sees through fir.declare (partial-entity access):
// a managed descriptor stays mapped even behind a declare.
func.func @managed_declare() {
  %0 = cuf.alloc !fir.box<!fir.ptr<!fir.array<?xf64>>> {bindc_name = "p", data_attr = #cuf.cuda<managed>, uniq_name = "_QFEp"} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>
  %1 = fir.declare %0 {uniq_name = "_QFEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>
  acc.parallel {
    %2 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_declare
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)

// -----

// A managed module global reached via fir.address_of must be mapped, not
// deviceptr. The walk follows the address-of edge to the global's attribute.
fir.global @mg_managed {data_attr = #cuf.cuda<managed>} : !fir.array<10xf64> {
  %0 = fir.zero_bits !fir.array<10xf64>
  fir.has_value %0 : !fir.array<10xf64>
}
func.func @managed_global() {
  %0 = fir.address_of(@mg_managed) : !fir.ref<!fir.array<10xf64>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_global
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf64>>) dataClause(acc_copy) implicit(true)

// -----

// A device module global is in device memory and stays deviceptr.
fir.global @mg_device {data_attr = #cuf.cuda<device>} : !fir.array<10xf64> {
  %0 = fir.zero_bits !fir.array<10xf64>
  fir.has_value %0 : !fir.array<10xf64>
}
func.func @device_global() {
  %0 = fir.address_of(@mg_device) : !fir.ref<!fir.array<10xf64>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @device_global
// CHECK-NOT: acc.copyin
// CHECK: acc.deviceptr varPtr(%{{.*}} : !fir.ref<!fir.array<10xf64>>) implicit(true)

// -----

// A managed dummy argument must be mapped, not deviceptr. The walk reads the
// attribute off the enclosing function argument.
func.func @managed_dummy(%arg0: !fir.ref<!fir.array<10xf64>> {cuf.data_attr = #cuf.cuda<managed>, fir.bindc_name = "a"}) {
  acc.parallel {
    %0 = fir.load %arg0 : !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_dummy
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf64>>) dataClause(acc_copy) implicit(true)

// -----

// A device dummy argument is in device memory and stays deviceptr.
func.func @device_dummy(%arg0: !fir.ref<!fir.array<10xf64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) {
  acc.parallel {
    %0 = fir.load %arg0 : !fir.ref<!fir.array<10xf64>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @device_dummy
// CHECK-NOT: acc.copyin
// CHECK: acc.deviceptr varPtr(%{{.*}} : !fir.ref<!fir.array<10xf64>>) implicit(true)

// -----

// A managed polymorphic (CLASS) POINTER descriptor must be mapped, not
// deviceptr -- same rule as the non-polymorphic descriptors.
func.func @managed_class_ptr() {
  %0 = cuf.alloc !fir.class<!fir.ptr<!fir.array<?xf64>>> {bindc_name = "cp", data_attr = #cuf.cuda<managed>, uniq_name = "_QFEcp"} -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?xf64>>>>
  acc.parallel {
    %1 = fir.load %0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xf64>>>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @managed_class_ptr
// CHECK-NOT: acc.deviceptr
// CHECK: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xf64>>>>) dataClause(acc_copy) implicit(true)
