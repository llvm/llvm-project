// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// Implicit data classification of CUDA data attributes for compute constructs.
//
// A `managed`/`unified` attribute establishes device *accessibility*, not device
// *residency*: the storage is reachable from the device but is not an already
// resident device object, so it must be mapped (so the runtime can attach and,
// where needed, privatize it) rather than treated as `deviceptr`. This holds for
// both POINTER (!fir.box<!fir.ptr>) and ALLOCATABLE (!fir.box<!fir.heap>)
// descriptors -- the classification keys on the attribute, not the type.
//
// `device` (and `constant`/`shared`) are genuinely device resident and stay
// device data, i.e. they are given `deviceptr`.

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

// A device descriptor is genuinely device resident and stays deviceptr.
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
