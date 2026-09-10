; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.mlir
; RUN: inter-opt %t.mlir '--inter-import-llvm=simd-width=32' -o %t.mlir
; RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner --compact %t.bin vadd 32 in:1 in:1000 out | FileCheck %s
; RUN: inter-runner %t.bin vadd 128 in:1 in:1000 out | %python %S/../../verify.py 'i*1+i*1000'

; CHECK: out0 = [0x00000000, 0x000003e9, 0x000007d2, 0x00000bbb, 0x00000fa4, 0x0000138d, 0x00001776, 0x00001b5f, 0x00001f48, 0x00002331, 0x0000271a, 0x00002b03, 0x00002eec, 0x000032d5, 0x000036be, 0x00003aa7, 0x00003e90, 0x00004279, 0x00004662, 0x00004a4b, 0x00004e34, 0x0000521d, 0x00005606, 0x000059ef, 0x00005dd8, 0x000061c1, 0x000065aa, 0x00006993, 0x00006d7c, 0x00007165, 0x0000754e, 0x00007937]

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @vadd(ptr addrspace(1) %a, ptr addrspace(1) %b,
                              ptr addrspace(1) %out) {
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %a.ptr = getelementptr i32, ptr addrspace(1) %a, i64 %gid
  %a.value = load i32, ptr addrspace(1) %a.ptr
  %b.ptr = getelementptr i32, ptr addrspace(1) %b, i64 %gid
  %b.value = load i32, ptr addrspace(1) %b.ptr
  %sum = add i32 %a.value, %b.value
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  store i32 %sum, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
