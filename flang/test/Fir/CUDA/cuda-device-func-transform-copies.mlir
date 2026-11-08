// Device copies made by cuf-duplicate-device-func take their original name back
// on the device; their host_device originals stay in the host module.

// RUN: fir-opt --cuf-transform-device-func %s | FileCheck %s
// RUN: fir-opt --cuf-duplicate-device-func --cuf-transform-device-func %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = ""} {

func.func @_QPhostdev(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = fir.load %arg0 : !fir.ref<i32>
  return
}
func.func @_QPhostdev.device(%arg0: !fir.ref<i32>) attributes {cuf.device_copy_of = @_QPhostdev, cuf.proc_attr = #cuf.cuda_proc<device>} {
  %0 = fir.load %arg0 : !fir.ref<i32>
  return
}

func.func @host_used_in_device() {
  return
}
func.func @host_used_in_device.device() attributes {cuf.device_copy_of = @host_used_in_device, cuf.proc_attr = #cuf.cuda_proc<device>} {
  return
}

func.func private @_QMotherPdecl() attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>}

func.func @_QPkernel(%arg0: !fir.ref<i32>) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  fir.call @_QPhostdev.device(%arg0) : (!fir.ref<i32>) -> ()
  fir.call @host_used_in_device.device() : () -> ()
  fir.call @_QMotherPdecl() : () -> ()
  %0 = fir.address_of(@host_used_in_device.device) : () -> ()
  return
}

func.func @_QPhostcaller(%arg0: !fir.ref<i32>) {
  fir.call @_QPhostdev(%arg0) : (!fir.ref<i32>) -> ()
  fir.call @host_used_in_device() : () -> ()
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%i : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    fir.call @_QPhostdev.device(%arg0) : (!fir.ref<i32>) -> ()
    "fir.end"() : () -> ()
  }
  return
}

}

// Host module: originals kept, copies gone.
// CHECK-LABEL: func.func @_QPhostdev(
// CHECK-SAME: cuf.proc_attr = #cuf.cuda_proc<host_device>
// CHECK: fir.load
// CHECK-NOT: func.func @_QPhostdev.device
// CHECK: func.func @host_used_in_device()
// CHECK-NOT: func.func @host_used_in_device.device
// CHECK: func.func private @_QMotherPdecl()

// The cuf.kernel region refers to the original name again.
// CHECK-LABEL: func.func @_QPhostcaller(
// CHECK: fir.call @_QPhostdev(
// CHECK: fir.call @host_used_in_device()
// CHECK: cuf.kernel
// CHECK: fir.call @_QPhostdev(

// GPU module: one function per original name, calls restored.
// CHECK: gpu.module @cuda_device_mod
// CHECK: gpu.func @_QPhostdev(
// CHECK: gpu.func @host_used_in_device()
// CHECK: func.func private @_QMotherPdecl()
// CHECK: gpu.func @_QPkernel(
// CHECK-SAME: kernel
// CHECK: fir.call @_QPhostdev(
// CHECK: fir.call @host_used_in_device()
// CHECK: fir.call @_QMotherPdecl()
// CHECK: fir.address_of(@host_used_in_device)

// The kernel keeps a host stub for registration, re-inserted after the module.
// CHECK: func.func @_QPkernel(
// CHECK-NEXT: return
// CHECK-NOT: .device
// CHECK-NOT: cuf.device_copy_of
