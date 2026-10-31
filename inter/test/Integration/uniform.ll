; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %s --import-llvm -o %t.mlir
; RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner --compact %t.bin uniform_kernel 32 out in:1 in:1000 u32:7 | FileCheck %s
; RUN: inter-runner %t.bin uniform_kernel 128 out in:1 in:1000 u32:7 | %python %S/../../verify.py 'i + 100 + 0*i'

; CHECK: out0 = [0x00000064, 0x00000065, 0x00000066, 0x00000067, 0x00000068, 0x00000069, 0x0000006a, 0x0000006b, 0x0000006c, 0x0000006d, 0x0000006e, 0x0000006f, 0x00000070, 0x00000071, 0x00000072, 0x00000073, 0x00000074, 0x00000075, 0x00000076, 0x00000077, 0x00000078, 0x00000079, 0x0000007a, 0x0000007b, 0x0000007c, 0x0000007d, 0x0000007e, 0x0000007f, 0x00000080, 0x00000081, 0x00000082, 0x00000083]

target datalayout = "e-i64:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @uniform_kernel(ptr addrspace(1) %out,
                                        ptr addrspace(1) %a,
                                        ptr addrspace(1) %b, i32 %threshold) {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %condition = icmp ugt i32 %threshold, 3
  br i1 %condition, label %then, label %else

then:
  %a.ptr = getelementptr i32, ptr addrspace(1) %a, i64 %gid
  %a.value = load i32, ptr addrspace(1) %a.ptr
  %then.value = add i32 %a.value, 100
  br label %merge

else:
  %b.ptr = getelementptr i32, ptr addrspace(1) %b, i64 %gid
  %else.value = load i32, ptr addrspace(1) %b.ptr
  br label %merge

merge:
  %result = phi i32 [ %then.value, %then ], [ %else.value, %else ]
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i64 %gid
  store i32 %result, ptr addrspace(1) %out.ptr
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
