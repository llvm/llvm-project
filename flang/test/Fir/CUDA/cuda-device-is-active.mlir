// RUN: fir-opt --split-input-file --cuf-convert-late %s | FileCheck %s
// RUN: fir-opt --split-input-file --canonicalize %s | FileCheck %s --check-prefix=CANON

// Host: lower to the CUDA Fortran runtime query.
func.func @host() -> i1 {
  %0 = cuf.device_is_active : i1
  return %0 : i1
}

// CHECK-LABEL: func.func @host
// CHECK: fir.call @_FortranACUFDeviceIsActive() : () -> i1
// CHECK-NOT: cuf.device_is_active

// -----

// Device copies in the GPU module must not call the host runtime.
gpu.module @cuda_device_mod {
  gpu.func @dev() -> i1 {
    %0 = cuf.device_is_active : i1
    gpu.return %0 : i1
  }
}

// CHECK-LABEL: gpu.func @dev
// CHECK: arith.constant false
// CHECK-NOT: fir.call @_FortranACUFDeviceIsActive
// CHECK-NOT: cuf.device_is_active

// -----

// Device procedures still in the host module are device context too.
func.func @device_proc() -> i1 attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %0 = cuf.device_is_active : i1
  return %0 : i1
}

// CHECK-LABEL: func.func @device_proc
// CHECK: arith.constant false
// CHECK-NOT: fir.call @_FortranACUFDeviceIsActive

// -----

// host_device is not a full device context; keep the runtime call on the host
// copy. The device clone lives in the GPU module and is folded separately.
func.func @host_device_proc() -> i1 attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %0 = cuf.device_is_active : i1
  return %0 : i1
}

// CHECK-LABEL: func.func @host_device_proc
// CHECK: fir.call @_FortranACUFDeviceIsActive() : () -> i1

// -----

// The query only reads context state: an unused one is dead and goes away,
// a used one stays.
func.func @unused_query() {
  %0 = cuf.device_is_active : i1
  return
}

// CANON-LABEL: func.func @unused_query
// CANON-NOT: cuf.device_is_active
// CANON: return

// -----

func.func @used_query() -> i1 {
  %0 = cuf.device_is_active : i1
  return %0 : i1
}

// CANON-LABEL: func.func @used_query
// CANON: %[[Q:.*]] = cuf.device_is_active : i1
// CANON: return %[[Q]]

// -----

// The motivating case: once the guarded cleanup is gone, the empty fir.if and
// the query that fed it are both dead.
func.func @empty_guard() {
  %0 = cuf.device_is_active : i1
  fir.if %0 {
  }
  return
}

// CANON-LABEL: func.func @empty_guard
// CANON-NOT: cuf.device_is_active
// CANON-NOT: fir.if
// CANON: return
