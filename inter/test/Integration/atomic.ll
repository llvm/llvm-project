; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.mlir
; RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner --compact --sort-output %t.bin atomic_kernel 32 out in:0 | FileCheck %s
; RUN: inter-runner %t.bin atomic_kernel 128 out in:0 | %python %S/../../verify.py --sort

; CHECK: out0 = [0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00000006, 0x00000007, 0x00000008, 0x00000009, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000d, 0x0000000e, 0x0000000f, 0x00000010, 0x00000011, 0x00000012, 0x00000013, 0x00000014, 0x00000015, 0x00000016, 0x00000017, 0x00000018, 0x00000019, 0x0000001a, 0x0000001b, 0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f]

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @atomic_kernel(ptr addrspace(1) %out,
                                       ptr addrspace(1) %counter) {
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %old = call spir_func i32 @_Z10atomic_addPU3AS1Vjj(
      ptr addrspace(1) %counter, i32 1)
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  store i32 %old, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i32 @_Z10atomic_addPU3AS1Vjj(ptr addrspace(1), i32)
