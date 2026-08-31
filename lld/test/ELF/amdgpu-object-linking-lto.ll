; REQUIRES: amdgpu

; Verify that the target symbol-finalization hook sees objects produced by LTO
; and resolves their SHN_AMDGPU_LDS symbols.

; RUN: llvm-as < %s -o %t.bc
; RUN: ld.lld --save-temps -shared -mllvm -amdgpu-enable-object-linking \
; RUN:   -mllvm -mcpu=gfx900 %t.bc -o %t
; RUN: llvm-readobj --symbols %t.lto.o | FileCheck %s --check-prefix=OBJECT
; RUN: llvm-readobj --symbols %t | FileCheck %s --check-prefix=LINKED

; OBJECT:      Name: lds
; OBJECT-NEXT: Value: 0x4
; OBJECT-NEXT: Size: 4
; OBJECT:      Section: Processor Specific (0xFF00)

; LINKED:      Name: lds
; LINKED-NEXT: Value: 0x0
; LINKED-NEXT: Size: 4
; LINKED:      Section: Absolute

target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

@lds = external addrspace(3) global i32, align 4

define amdgpu_kernel void @kernel() {
entry:
  store volatile i32 0, ptr addrspace(3) @lds, align 4
  ret void
}
