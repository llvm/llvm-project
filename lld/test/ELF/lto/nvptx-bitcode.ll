; REQUIRES: nvptx

; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld --lto-emit-asm --plugin-opt=-mcpu=sm_70 --plugin-opt=-mattr=+ptx70 %t.bc -o %t
;
; lld writes assembly output produced by LTO to a .lto.s side file.
; RUN: FileCheck %s --check-prefix=PTX < %t.lto.s

; RUN: ld.lld --lto-emit-llvm %t.bc -o %t.lto.bc
; RUN: llvm-dis %t.lto.bc -o - | FileCheck %s --check-prefix=LLVM

; RUN: not ld.lld %t.bc -o %t.out 2>&1 | FileCheck %s --check-prefix=ERR

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; PTX: .version 7.0
; PTX: .target sm_70
; PTX: .address_size 64

; LLVM: target triple = "nvptx64-nvidia-cuda"

; ERR: error: CUDA bitcode inputs are only supported with --lto-emit-asm or --lto-emit-llvm
