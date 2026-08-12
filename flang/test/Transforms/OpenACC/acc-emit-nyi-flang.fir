// RUN: fir-opt %s --acc-emit-nyi-flang --verify-diagnostics

func.func @acc_func() attributes {
  acc.routine_info = #acc.routine_info<[@acc_func_routine]>
} {
  return
}

// expected-error @+1 {{not yet implemented: OpenACC routine directive}}
acc.routine @acc_func_routine func(@acc_func)

func.func @directives(%ptr: !llvm.ptr, %v: memref<i32>,
                      %x: memref<i32>, %value: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // expected-error @+1 {{not yet implemented: OpenACC parallel directive}}
  acc.parallel {
    acc.yield
  }
  // expected-error @+1 {{not yet implemented: OpenACC kernels directive}}
  acc.kernels {
    acc.terminator
  }
  // expected-error @+1 {{not yet implemented: OpenACC serial directive}}
  acc.serial {
    acc.yield
  }

  // expected-error @+1 {{not yet implemented: OpenACC parallel loop directive}}
  acc.parallel combined(loop) {
    // expected-error @+1 {{not yet implemented: OpenACC loop directive}}
    acc.loop combined(parallel) control(%iv : index) = (%c0 : index)
        to (%c1 : index) step (%c1 : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.terminator
  }
  // expected-error @+1 {{not yet implemented: OpenACC kernels loop directive}}
  acc.kernels combined(loop) {
    // expected-error @+1 {{not yet implemented: OpenACC loop directive}}
    acc.loop combined(kernels) control(%iv : index) = (%c0 : index)
        to (%c1 : index) step (%c1 : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.terminator
  }
  // expected-error @+1 {{not yet implemented: OpenACC serial loop directive}}
  acc.serial combined(loop) {
    // expected-error @+1 {{not yet implemented: OpenACC loop directive}}
    acc.loop combined(serial) control(%iv : index) = (%c0 : index)
        to (%c1 : index) step (%c1 : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.terminator
  }

  // expected-error @+1 {{not yet implemented: OpenACC data directive}}
  acc.data {
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue none>}
  // expected-error @+1 {{not yet implemented: OpenACC loop directive}}
  acc.loop control(%iv : index) = (%c0 : index) to (%c1 : index)
      step (%c1 : index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}

  // expected-error @+1 {{not yet implemented: OpenACC operation acc.copyin}}
  %copyin = acc.copyin varPtr(%ptr : !llvm.ptr) varType(i32) -> !llvm.ptr
  // expected-error @+1 {{not yet implemented: OpenACC enter data directive}}
  acc.enter_data dataOperands(%copyin : !llvm.ptr)
  // expected-error @+1 {{not yet implemented: OpenACC operation acc.getdeviceptr}}
  %devicePtr = acc.getdeviceptr varPtr(%ptr : !llvm.ptr) varType(i32) -> !llvm.ptr
  // expected-error @+1 {{not yet implemented: OpenACC exit data directive}}
  acc.exit_data dataOperands(%devicePtr : !llvm.ptr)

  // expected-error @+1 {{not yet implemented: OpenACC operation acc.use_device}}
  %useDevice = acc.use_device varPtr(%ptr : !llvm.ptr) varType(i32) -> !llvm.ptr
  // expected-error @+1 {{not yet implemented: OpenACC host_data directive}}
  acc.host_data dataOperands(%useDevice : !llvm.ptr) {
    acc.terminator
  }

  // expected-error @+1 {{not yet implemented: OpenACC init directive}}
  acc.init
  // expected-error @+1 {{not yet implemented: OpenACC shutdown directive}}
  acc.shutdown
  // expected-error @+1 {{not yet implemented: OpenACC set directive}}
  acc.set attributes {device_type = #acc.device_type<nvidia>}
  // expected-error @+1 {{not yet implemented: OpenACC wait directive}}
  acc.wait

  // expected-error @+1 {{not yet implemented: OpenACC operation acc.update_device}}
  %updateDevice = acc.update_device varPtr(%ptr : !llvm.ptr) varType(i32) -> !llvm.ptr
  // expected-error @+1 {{not yet implemented: OpenACC update directive}}
  acc.update dataOperands(%updateDevice : !llvm.ptr)

  // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
  acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
  // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
  acc.atomic.write %x = %value : memref<i32>, i32
  // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%old: i32):
    acc.yield %old : i32
  }
  // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
  acc.atomic.capture {
    // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    // expected-error @+1 {{not yet implemented: OpenACC atomic directive}}
    acc.atomic.write %x = %value : memref<i32>, i32
  }

  // expected-error @+1 {{not yet implemented: OpenACC cache directive}}
  %cache = acc.cache varPtr(%ptr : !llvm.ptr) varType(i32) -> !llvm.ptr

  // expected-error @+1 {{not yet implemented: OpenACC declare directive}}
  acc.declare_enter dataOperands(%copyin : !llvm.ptr)
  // expected-error @+1 {{not yet implemented: OpenACC declare directive}}
  acc.declare_exit dataOperands(%copyin : !llvm.ptr)
  // expected-error @+1 {{not yet implemented: OpenACC declare directive}}
  acc.declare dataOperands(%copyin : !llvm.ptr) {
    acc.terminator
  }

  return
}
