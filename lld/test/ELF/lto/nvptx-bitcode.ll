; REQUIRES: nvptx
;; NVPTX bitcode can be compiled with --lto-emit-asm or --lto-emit-llvm.
;; Kernels must be exported to survive internalization.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as a.ll -o a.bc
; RUN: llvm-as b.ll -o b.bc

; RUN: ld.lld --export-dynamic-symbol=kern --lto-emit-asm --plugin-opt=mcpu=sm_90 a.bc b.bc -o out
; RUN: FileCheck %s --check-prefix=PTX < out.lto.s

; PTX:      .version 7.8
; PTX-NEXT: .target sm_90
; PTX-NEXT: .address_size 64
; PTX-NOT:  .func
; PTX:      .visible .entry kern(
; PTX-NOT:  .func

; RUN: ld.lld --export-dynamic-symbol=kern --lto-emit-llvm a.bc b.bc -o out.bc
; RUN: llvm-dis out.bc -o - | FileCheck %s --check-prefix=LLVM

; LLVM:     target triple = "nvptx64-nvidia-cuda"
; LLVM:     define dso_local ptx_kernel void @kern(
; LLVM-NOT: @unused

; RUN: not ld.lld a.bc b.bc 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

; ERR: error: CUDA input is only supported with --lto-emit-asm or --lto-emit-llvm

;--- a.ll
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @helper(ptr)

define ptx_kernel void @kern(ptr %p) {
  call void @helper(ptr %p)
  ret void
}

;--- b.ll
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @helper(ptr %p) {
  store i32 1, ptr %p
  ret void
}

define void @unused() {
  ret void
}
