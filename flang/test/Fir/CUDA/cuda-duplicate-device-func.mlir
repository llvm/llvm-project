// RUN: fir-opt --split-input-file --cuf-duplicate-device-func %s | FileCheck %s

module attributes {fir.allocation_policy = #fir.allocation_policy<stack_arrays = true, small_array_threshold = 1024, total_stack_limit = 4194304>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = ""} {

// A host_device procedure: the original stays host code, the copy is device
// code and opts out of -fstack-arrays.
func.func @_QPhostdev(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = fir.load %arg0 : !fir.ref<i32>
  fir.call @deeper() : () -> ()
  return
}

// Reached only through the host_device procedure above: copied as well, and
// the copy of the caller refers to it.
func.func @deeper() {
  return
}

// Reached only through a procedure reference in device code.
func.func @by_address() {
  return
}

// Called directly by device code.
func.func @host_used_in_device() {
  return
}

// Declarations are not copied; device code keeps calling them by name.
func.func private @_QMotherPdecl() attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>}
func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1 attributes {fir.runtime}

// Not duplicated: already device code.
func.func @_QPdevonly() attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  return
}

// Not duplicated: OpenACC routines are moved by the OpenACC pipeline.
func.func @acc_routine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_info]>} {
  return
}

// Device code: every reference to a copied procedure is redirected.
func.func @_QPkernel(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i8>, %arg2: !fir.box<none>) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  fir.call @_QPhostdev(%arg0) : (!fir.ref<i32>) -> ()
  fir.call @host_used_in_device() : () -> ()
  %0 = fir.address_of(@by_address) : () -> ()
  fir.call @_QMotherPdecl() : () -> ()
  %1 = fir.call @_FortranAioOutputDescriptor(%arg1, %arg2) : (!fir.ref<i8>, !fir.box<none>) -> i1
  fir.call @_QPdevonly() : () -> ()
  fir.call @acc_routine() : () -> ()
  return
}

// Host code: references keep the originals.
func.func @_QPhostcaller(%arg0: !fir.ref<i32>) {
  fir.call @_QPhostdev(%arg0) : (!fir.ref<i32>) -> ()
  fir.call @host_used_in_device() : () -> ()
  %0 = fir.address_of(@by_address) : () -> ()
  return
}

}

// CHECK-LABEL: func.func @_QPhostdev(
// CHECK-SAME: attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>}
// CHECK: fir.call @deeper()

// CHECK: func.func @_QPhostdev.device(
// CHECK-SAME: cuf.device_copy_of = @_QPhostdev
// CHECK-SAME: cuf.proc_attr = #cuf.cuda_proc<device>
// CHECK-SAME: fir.allocation_policy = #fir.allocation_policy<stack_arrays = false
// CHECK: fir.load
// CHECK: fir.call @deeper.device()

// CHECK: func.func @deeper()
// CHECK-NOT: cuf.device_copy_of
// CHECK: func.func @deeper.device()
// CHECK-SAME: cuf.device_copy_of = @deeper

// CHECK: func.func @by_address()
// CHECK: func.func @by_address.device()
// CHECK-SAME: cuf.device_copy_of = @by_address

// CHECK: func.func @host_used_in_device()
// CHECK: func.func @host_used_in_device.device()
// CHECK-SAME: cuf.device_copy_of = @host_used_in_device

// CHECK: func.func private @_QMotherPdecl()
// CHECK-NOT: @_QMotherPdecl.device
// CHECK: func.func private @_FortranAioOutputDescriptor(
// CHECK-NOT: @_FortranAioOutputDescriptor.device

// CHECK: func.func @_QPdevonly()
// CHECK-NOT: @_QPdevonly.device
// CHECK: func.func @acc_routine()
// CHECK-NOT: @acc_routine.device

// CHECK-LABEL: func.func @_QPkernel(
// CHECK: fir.call @_QPhostdev.device(
// CHECK: fir.call @host_used_in_device.device()
// CHECK: fir.address_of(@by_address.device)
// CHECK: fir.call @_QMotherPdecl()
// CHECK: fir.call @_FortranAioOutputDescriptor(
// CHECK: fir.call @_QPdevonly()
// CHECK: fir.call @acc_routine()

// CHECK-LABEL: func.func @_QPhostcaller(
// CHECK: fir.call @_QPhostdev(
// CHECK: fir.call @host_used_in_device()
// CHECK: fir.address_of(@by_address)

// -----

// Three host_device procedures calling each other. Two separate chains come
// out: the copies call the copies and the originals keep calling the originals.

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = ""} {

func.func @_QPlevel3(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = fir.load %arg0 : !fir.ref<i32>
  return
}

func.func @_QPlevel2(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  fir.call @_QPlevel3(%arg0) : (!fir.ref<i32>) -> ()
  return
}

func.func @_QPlevel1(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  fir.call @_QPlevel2(%arg0) : (!fir.ref<i32>) -> ()
  return
}

func.func @_QPkernel(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  fir.call @_QPlevel1(%arg0) : (!fir.ref<i32>) -> ()
  return
}

func.func @_QPhostcaller(%arg0: !fir.ref<i32>) {
  fir.call @_QPlevel1(%arg0) : (!fir.ref<i32>) -> ()
  return
}

}

// The original chain, untouched.
// CHECK-LABEL: func.func @_QPlevel3(
// CHECK-SAME: cuf.proc_attr = #cuf.cuda_proc<host_device>
// CHECK: fir.load
// CHECK-LABEL: func.func @_QPlevel3.device(
// CHECK-SAME: cuf.device_copy_of = @_QPlevel3
// CHECK: fir.load

// CHECK-LABEL: func.func @_QPlevel2(
// CHECK-SAME: cuf.proc_attr = #cuf.cuda_proc<host_device>
// CHECK: fir.call @_QPlevel3(
// CHECK-NOT: .device
// CHECK-LABEL: func.func @_QPlevel2.device(
// CHECK-SAME: cuf.device_copy_of = @_QPlevel2
// CHECK: fir.call @_QPlevel3.device(

// CHECK-LABEL: func.func @_QPlevel1(
// CHECK-SAME: cuf.proc_attr = #cuf.cuda_proc<host_device>
// CHECK: fir.call @_QPlevel2(
// CHECK-NOT: .device
// CHECK-LABEL: func.func @_QPlevel1.device(
// CHECK-SAME: cuf.device_copy_of = @_QPlevel1
// CHECK: fir.call @_QPlevel2.device(

// Device code enters the copy chain, host code the original one.
// CHECK-LABEL: func.func @_QPkernel(
// CHECK: fir.call @_QPlevel1.device(
// CHECK-LABEL: func.func @_QPhostcaller(
// CHECK: fir.call @_QPlevel1(
// CHECK-NOT: .device
