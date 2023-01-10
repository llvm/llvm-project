; REQUIRES: amdgpu

; TODO(slinder1): Workaround for HeterogeneousDWARF to support `-fgpu-rdc
; -O0 -g`. Remove this when we support higher optimization levels.

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -plugin-opt=O0 -plugin-opt=mcpu=gfx90a %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=CHECK-O0 %s
; RUN: ld.lld -plugin-opt=O1 -plugin-opt=mcpu=gfx90a %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=CHECK-O1 %s
; RUN: ld.lld -plugin-opt=O2 -plugin-opt=mcpu=gfx90a %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=CHECK-O2 %s
; RUN: ld.lld -plugin-opt=O3 -plugin-opt=mcpu=gfx90a %t.o -o %t -mllvm -debug-pass=Structure 2>&1 | FileCheck --check-prefix=CHECK-O3 %s

; CHECK-O0: Fast Register Allocator
; CHECK-O1: Greedy Register Allocator
; CHECK-O2: Greedy Register Allocator
; CHECK-O3: Greedy Register Allocator

target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @_start() {
entry:
  ret void
}
