// RUN: fir-opt %s -acc-initialize-fir-analyses -acc-routine-to-gpu-func -split-input-file | FileCheck %s

// CHECK: gpu.module @acc_gpu_module
// CHECK: gpu.func @_QPfoo
// CHECK: fir.store
// CHECK: gpu.return
acc.routine @routine_seq func(@_QPfoo) seq
func.func @_QPfoo(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) attributes {acc.specialized_routine = #acc.specialized_routine<@routine_seq, <seq>, "_QPfoo">} {
  %c0 = arith.constant 0 : i32
  fir.store %c0 to %arg0 : !fir.ref<i32>
  return
}

// -----

// One routine with body, one declaration-only; both end up in GPU module.
// CHECK: acc.routine @acc_routine_0
// CHECK: acc.routine @acc_routine_1
// CHECK: gpu.module @{{.*}}
// CHECK-NEXT: gpu.func @devicefunc(){{.*}} {
// CHECK-NEXT: gpu.return
// CHECK-NEXT: }
// CHECK-NEXT: func.func private @declfunc()
module {
  acc.routine @acc_routine_0 func(@devicefunc)
  func.func @devicefunc() attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <seq>, "devicefunc">} {
    return
  }
  acc.routine @acc_routine_1 func(@declfunc)
  func.func private @declfunc() -> ()
}

// -----

// nohost routine: host copy is removed after moving to GPU module.
// CHECK: acc.routine @acc_routine_0
// CHECK-NOT: func.func @_QMmPfv_0
module {
  acc.routine @acc_routine_0 func(@_QMmPfv_0) vector nohost
  func.func @_QMmPfv_0(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <vector>, "_QMmPfv_0">} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QMmFfvEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    return
  }
}

// -----

// Routine that calls another acc routine: both end up in GPU module; call kept.
// CHECK: acc.routine @r_outer
// CHECK: acc.routine @r_inner
// CHECK: gpu.module @{{.*}}
// CHECK: gpu.func @outer
// CHECK: fir.call @inner
// CHECK: gpu.return
// CHECK: gpu.func @inner
module {
  acc.routine @r_outer func(@outer) seq
  acc.routine @r_inner func(@inner) seq
  func.func @outer() attributes {acc.specialized_routine = #acc.specialized_routine<@r_outer, <seq>, "outer">} {
    fir.call @inner() : () -> ()
    return
  }
  func.func @inner() attributes {acc.specialized_routine = #acc.specialized_routine<@r_inner, <seq>, "inner">} {
    return
  }
}

// -----

// Routine calls a Fortran runtime function (fir.runtime): valid for device, decl in GPU module.
// CHECK: acc.routine @r_seq
// CHECK: gpu.module @{{.*}}
// CHECK: gpu.func @_QPcalls_runtime
// CHECK: fir.call @_FortranASomeRuntime
// CHECK: gpu.return
// CHECK: func.func private @_FortranASomeRuntime
module {
  acc.routine @r_seq func(@_QPcalls_runtime) seq
  func.func @_QPcalls_runtime() attributes {acc.specialized_routine = #acc.specialized_routine<@r_seq, <seq>, "_QPcalls_runtime">} {
    fir.call @_FortranASomeRuntime() : () -> ()
    return
  }
  func.func private @_FortranASomeRuntime() -> () attributes {fir.runtime}
}

// -----

// Routine calls a CUF device procedure (cuf.proc_attr = device): valid for device, decl in GPU module.
// CHECK: acc.routine @r_seq
// CHECK: gpu.module @{{.*}}
// CHECK: gpu.func @_QPcalls_cuf
// CHECK: fir.call @_QPcuf_device
// CHECK: gpu.return
// CHECK: func.func private @_QPcuf_device
module {
  acc.routine @r_seq func(@_QPcalls_cuf) seq
  func.func @_QPcalls_cuf() attributes {acc.specialized_routine = #acc.specialized_routine<@r_seq, <seq>, "_QPcalls_cuf">} {
    fir.call @_QPcuf_device() : () -> ()
    return
  }
  func.func private @_QPcuf_device() -> () attributes {cuf.proc_attr = #cuf.cuda_proc<device>}
}
