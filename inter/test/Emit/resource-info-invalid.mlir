// RUN: not inter-translate %s --xemachine-to-zebin -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @stale() attributes {
      xemachine.barrier_count = 0 : i32,
      xemachine.grf_count = 128 : i32,
      xemachine.grf_used = 1 : i32,
      xemachine.has_dpas = false,
      xemachine.has_global_atomics = false,
      xemachine.has_no_stateless_write = true,
      xemachine.kernel_args = [],
      xemachine.simd_size = 32 : i32,
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %data = xemachine.archreg 8 : !xemachine.reg<32, 8>
    return
  }
}

// CHECK: stale machine resource attributes
