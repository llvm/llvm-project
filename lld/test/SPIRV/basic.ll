; REQUIRES: spirv, spirv-tools
; RUN: llvm-as %s -o %t_input.bc
; RUN: ld.lld -flavor spirv %t_input.bc -o %t_output.spv

; RUN: spirv-lld %t_input.bc -o %t_output2.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @test(i32 addrspace(1)* %ptr) {
  store i32 42, i32 addrspace(1)* %ptr
  ret void
}
