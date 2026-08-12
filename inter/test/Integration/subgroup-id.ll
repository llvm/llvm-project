; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.mlir
; RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner --group-size 256 %t.bin subgroup_id 256 out | %python %S/../../verify.py 'i//16'

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @subgroup_id(ptr addrspace(1) %out) {
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %subgroup = call spir_func i32 @_Z16get_sub_group_idv()
  %gid.i32 = trunc i64 %gid to i32
  %lane.zero = mul i32 %gid.i32, 0
  %value = add i32 %subgroup, %lane.zero
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  store i32 %value, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i32 @_Z16get_sub_group_idv()
