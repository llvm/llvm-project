// RUN: inter-opt %S/Inputs/lighthouse-matmul-xegpu-wg.mlir | FileCheck %s

// CHECK-LABEL: gpu.func @payload_kernel
// CHECK: xegpu.prefetch_nd
// CHECK: xegpu.load_nd
// CHECK: xegpu.dpas
// CHECK: xegpu.store_nd
