; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.mlir
; RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner --compact %t.bin slm_kernel 32 out in:1 | FileCheck %s
; RUN: inter-runner %t.bin slm_kernel 128 out in:1 | %python %S/../../verify.py 'i + (i & ~31) + 31 - (i & 31)'

; CHECK: out0 = [0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f]

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

@slm_kernel.tile = internal addrspace(3) global [32 x i32] undef, align 4

define spir_kernel void @slm_kernel(ptr addrspace(1) %out,
                                    ptr addrspace(1) %in) {
  %lid64 = call spir_func i64 @_Z12get_local_idj(i32 0)
  %lid = trunc i64 %lid64 to i32
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %in.ptr = getelementptr i32, ptr addrspace(1) %in, i64 %gid
  %value = load i32, ptr addrspace(1) %in.ptr
  %tile.ptr = getelementptr [32 x i32], ptr addrspace(3) @slm_kernel.tile,
                            i64 0, i64 %lid64
  store i32 %value, ptr addrspace(3) %tile.ptr
  call spir_func void @_Z7barrierj(i32 1)
  %reverse.lid = sub i32 31, %lid
  %reverse.lid64 = zext i32 %reverse.lid to i64
  %reverse.ptr = getelementptr [32 x i32], ptr addrspace(3) @slm_kernel.tile,
                               i64 0, i64 %reverse.lid64
  %reverse.value = load i32, ptr addrspace(3) %reverse.ptr
  %result = add i32 %value, %reverse.value
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  store i32 %result, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z12get_local_idj(i32)
declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func void @_Z7barrierj(i32)
