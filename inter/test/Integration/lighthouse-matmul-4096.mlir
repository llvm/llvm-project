// REQUIRES: host-supports-inter-bmg
// RUN: %python %S/Inputs/generate-lighthouse-shape.py %S/Inputs/lighthouse-matmul.mlir %t.input.mlir 4096 4096
// RUN: inter-opt %t.input.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
// RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
// RUN: inter-matmul-runner %t.bin 4096 4096 | FileCheck %s

// CHECK: PASS: 4096x4096x4096 structured-random f16 matmul, max error 0
