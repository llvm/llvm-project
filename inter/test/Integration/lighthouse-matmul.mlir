// REQUIRES: host-supports-inter-bmg
// RUN: inter-opt %S/Inputs/lighthouse-matmul.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
// RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
// RUN: inter-matmul-runner %t.bin | FileCheck %s

// CHECK: PASS: 128x128x64 random f16 matmul, max error 0
