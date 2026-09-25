// RUN: fir-opt --cuf-device-global %s | FileCheck %s

// Mutually referencing type-descriptor globals must not cause infinite
// recursion. One GPU copy must get a zero initializer to break the cycle,
// while the other keeps its complete initializer. Both remain definitions.
module attributes {gpu.container_module} {
  fir.global linkonce @_QMmE.dt.t1 constant : i64 {
    %0 = fir.address_of(@_QMmE.dt.t2) : !fir.ref<i64>
    %1 = fir.convert %0 : (!fir.ref<i64>) -> i64
    fir.has_value %1 : i64
  }
  fir.global linkonce @_QMmE.dt.t2 constant : i64 {
    %0 = fir.address_of(@_QMmE.dt.t1) : !fir.ref<i64>
    %1 = fir.convert %0 : (!fir.ref<i64>) -> i64
    fir.has_value %1 : i64
  }

  func.func @_QMmPdev() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %0 = fir.alloca !fir.type<_QMmTt1{i:i32}>
    return
  }
}

// Host copies keep the cycle.
// CHECK: fir.global linkonce @_QMmE.dt.t1
// CHECK: fir.address_of(@_QMmE.dt.t2)
// CHECK: fir.global linkonce @_QMmE.dt.t2
// CHECK: fir.address_of(@_QMmE.dt.t1)

// Cut the source of the back edge; replace its init with zero_bits.
// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK-DAG: fir.global linkonce @_QMmE.dt.t2 constant : i64
// CHECK-DAG: fir.zero_bits
// CHECK-DAG: fir.has_value
// CHECK-DAG: fir.global linkonce @_QMmE.dt.t1 constant : i64
// CHECK-DAG: fir.address_of(@_QMmE.dt.t2)
// CHECK-DAG: fir.has_value
