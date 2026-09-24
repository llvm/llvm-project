// RUN: fir-opt --split-input-file --cuf-convert-late %s | FileCheck %s

// Host: lower to the CUDA Fortran runtime query.
func.func @host() {
  %0 = cuf.device_is_active : i1
  return
}

// CHECK-LABEL: func.func @host
// CHECK: fir.call @_FortranACUFDeviceIsActive() : () -> i1
// CHECK-NOT: cuf.device_is_active

// -----

// Device copies in the GPU module must not call the host runtime.
gpu.module @cuda_device_mod {
  gpu.func @dev() {
    %0 = cuf.device_is_active : i1
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @dev
// CHECK: arith.constant false
// CHECK-NOT: fir.call @_FortranACUFDeviceIsActive
// CHECK-NOT: cuf.device_is_active

// -----

// Device procedures still in the host module are device context too.
func.func @device_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %0 = cuf.device_is_active : i1
  return
}

// CHECK-LABEL: func.func @device_proc
// CHECK: arith.constant false
// CHECK-NOT: fir.call @_FortranACUFDeviceIsActive

// -----

// host_device is not a full device context; keep the runtime call on the host
// copy. The device clone lives in the GPU module and is folded separately.
func.func @host_device_proc() attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = cuf.device_is_active : i1
  return
}

// CHECK-LABEL: func.func @host_device_proc
// CHECK: fir.call @_FortranACUFDeviceIsActive() : () -> i1
