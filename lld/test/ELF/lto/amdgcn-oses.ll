; REQUIRES: amdgpu

; RUN: split-file %s %t

; RUN: llvm-as %t/amdhsa.ll -o %t/amdhsa.o
; RUN: ld.lld %t/amdhsa.o -o %t/amdhsa.so
; RUN: llvm-readobj --file-headers %t/amdhsa.so | FileCheck %s --check-prefixes=GCN,AMDHSA

; RUN: llvm-as %t/amdpal.ll -o %t/amdpal.o
; RUN: ld.lld %t/amdpal.o -o %t/amdpal.so
; RUN: llvm-readobj --file-headers %t/amdpal.so | FileCheck %s --check-prefixes=GCN,NON-AMDHSA,AMDPAL

; RUN: llvm-as %t/mesa3d.ll -o %t/mesa3d.o
; RUN: ld.lld %t/mesa3d.o -o %t/mesa3d.so
; RUN: llvm-readobj --file-headers %t/mesa3d.so | FileCheck %s --check-prefixes=GCN,NON-AMDHSA,MESA3D

; AMDHSA: OS/ABI: AMDGPU_HSA (0x40)
; AMDHSA: ABIVersion: 2

; AMDPAL: OS/ABI: AMDGPU_PAL (0x41)
; MESA3D: OS/ABI: AMDGPU_MESA3D (0x42)
; NON-AMDHSA: ABIVersion: 0

; GCN: Machine: EM_AMDGPU

;--- amdhsa.ll
target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @_start() {
  ret void
}

;--- amdpal.ll
target triple = "amdgcn-amd-amdpal"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define amdgpu_cs void @_start() {
  ret void
}

;--- mesa3d.ll
target triple = "amdgcn-amd-mesa3d"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @_start() {
  ret void
}
