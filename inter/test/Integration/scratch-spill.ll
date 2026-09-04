; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.frontend.mlir
; RUN: inter-opt %t.frontend.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_lower_to_machine})' -o %t.machine.mlir
; RUN: inter-opt %t.machine.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%S/Inputs/scratch-pressure.mlir},transform-interpreter{entry-point=set_scratch_pressure})' -o %t.pressure.mlir
; RUN: inter-opt %t.pressure.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc},func.func(inter-insert-sync,inter-resource-info))' -o %t.mlir
; RUN: FileCheck %s < %t.mlir
; RUN: inter-translate %t.mlir --xemachine-to-ged -o %t.ged
; RUN: inter-ged-dump %t.ged | FileCheck %s --check-prefix=GED
; RUN: inter-translate %t.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner %t.bin scratch_spill 128 inout:1 in:10 in:100 in:1000 | %python %S/../../verify.py 'i*1118'

; CHECK: xemachine.scratch_size
; CHECK: xemachine.scratch_access

; GED: opcode=and exec=1 swsb=0x0 {{.*}}dst=grf[[SSO:[0-9]+]].0:ud<1> src0=grf0.20:ud
; GED: opcode=shr exec=1 swsb=0x11 {{.*}}dst=arf16.8:ud<1> src0=grf[[SSO]].0:ud

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @scratch_spill(ptr addrspace(1) %out,
                                       ptr addrspace(1) %b,
                                       ptr addrspace(1) %c,
                                       ptr addrspace(1) %d) {
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  %out.value = load i32, ptr addrspace(1) %out.ptr
  %biased = mul i32 %out.value, 3
  %weighted = mul i32 %out.value, 5
  %b.ptr = getelementptr i32, ptr addrspace(1) %b, i64 %gid
  %b.value = load i32, ptr addrspace(1) %b.ptr
  %c.ptr = getelementptr i32, ptr addrspace(1) %c, i64 %gid
  %c.value = load i32, ptr addrspace(1) %c.ptr
  %d.ptr = getelementptr i32, ptr addrspace(1) %d, i64 %gid
  %d.value = load i32, ptr addrspace(1) %d.ptr
  %bc = add i32 %b.value, %c.value
  %bcd = add i32 %bc, %d.value
  %bias = add i32 %biased, %weighted
  %sum = add i32 %bias, %bcd
  store i32 %sum, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
