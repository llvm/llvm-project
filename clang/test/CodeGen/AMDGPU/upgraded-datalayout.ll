; REQUIRES: amdgpu-registered-target

; RUN: split-file %s %t

; The data layout string of old-datalayout.ll comes from an older LLVM. When
; clang reads the bitcode, it upgrades the string. The upgraded string
; describes the target layout, but it lists the specifications in a different
; order. Clang must accept it.
; RUN: llvm-as %t/old-datalayout.ll -o %t/old-datalayout.bc
; RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -x ir %t/old-datalayout.bc -emit-obj -o /dev/null

; A layout that differs from the target layout is still an error.
; RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -x ir %t/wrong-datalayout.ll -emit-llvm-bc -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: backend data layout '{{.*}}-p3:64:64-{{.*}}' does not match expected target description '{{.*}}-p3:32:32-{{.*}}'

;--- old-datalayout.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @func() {
  ret void
}

;--- wrong-datalayout.ll
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:64:64-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @func() {
  ret void
}
