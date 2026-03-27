// RUN: fir-opt --split-input-file --cuf-transform-device-func="compute-capability=90" %s | FileCheck %s

func.func @_QPsub_maxtnid() attributes {cuf.launch_bounds = #cuf.launch_bounds<maxTPB = 256 : i64, minBPM = 2 : i64, upperBoundClusterSize = 3 : i64>, cuf.proc_attr = #cuf.cuda_proc<global>} {
  %cst = arith.constant 2.000000e+00 : f32
  return
}

// CHECK: gpu.func @_QPsub_maxtnid() kernel attributes {nvvm.cluster_max_blocks = 3 : i64, nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i64}
