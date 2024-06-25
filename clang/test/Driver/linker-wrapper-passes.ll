; Check various clang-linker-wrapper pass options after -offload-opt.

; REQUIRES: llvm-plugins, llvm-examples
; REQUIRES: x86-registered-target
; REQUIRES: amdgpu-registered-target

; Setup.
; RUN: split-file %s %t
; RUN: opt -o %t/host-x86_64-unknown-linux-gnu.bc \
; RUN:     %t/host-x86_64-unknown-linux-gnu.ll
; RUN: opt -o %t/openmp-amdgcn-amd-amdhsa.bc \
; RUN:     %t/openmp-amdgcn-amd-amdhsa.ll
; RUN: clang-offload-packager -o %t/openmp-x86_64-unknown-linux-gnu.out \
; RUN:     --image=file=%t/openmp-amdgcn-amd-amdhsa.bc,triple=amdgcn-amd-amdhsa
; RUN: %clang -cc1 -S -o %t/host-x86_64-unknown-linux-gnu.s \
; RUN:     -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
; RUN:     -fembed-offload-object=%t/openmp-x86_64-unknown-linux-gnu.out \
; RUN:     %t/host-x86_64-unknown-linux-gnu.bc
; RUN: %clang -cc1as -o %t/host-x86_64-unknown-linux-gnu.o \
; RUN:     -triple x86_64-unknown-linux-gnu -filetype obj -target-cpu x86-64 \
; RUN:     %t/host-x86_64-unknown-linux-gnu.s

; Check plugin, -passes, and no remarks.
; RUN: clang-linker-wrapper -o a.out --embed-bitcode \
; RUN:     --linker-path=/usr/bin/true %t/host-x86_64-unknown-linux-gnu.o \
; RUN:     %offload-opt-loadbye --offload-opt=-wave-goodbye \
; RUN:     --offload-opt=-passes="function(goodbye),module(inline)" 2>&1 | \
; RUN:   FileCheck -match-full-lines -check-prefixes=OUT %s

; Check plugin, -p, and remarks.
; RUN: clang-linker-wrapper -o a.out --embed-bitcode \
; RUN:     --linker-path=/usr/bin/true %t/host-x86_64-unknown-linux-gnu.o \
; RUN:     %offload-opt-loadbye --offload-opt=-wave-goodbye \
; RUN:     --offload-opt=-p="function(goodbye),module(inline)" \
; RUN:     --offload-opt=-pass-remarks=inline \
; RUN:     --offload-opt=-pass-remarks-output=%t/remarks.yml \
; RUN:     --offload-opt=-pass-remarks-filter=inline \
; RUN:     --offload-opt=-pass-remarks-format=yaml 2>&1 | \
; RUN:   FileCheck -match-full-lines -check-prefixes=OUT,REM %s
; RUN: FileCheck -input-file=%t/remarks.yml -match-full-lines \
; RUN:     -check-prefixes=YML %s

; Check handling of bad plugin.
; RUN: not clang-linker-wrapper \
; RUN:     --offload-opt=-load-pass-plugin=%t/nonexistent.so 2>&1 | \
; RUN:   FileCheck -match-full-lines -check-prefixes=BAD-PLUGIN %s

;  OUT-NOT: {{.}}
;      OUT: Bye: f
; OUT-NEXT: Bye: test
; REM-NEXT: remark: {{.*}} 'f' inlined into 'test' {{.*}}
;  OUT-NOT: {{.}}

;  YML-NOT: {{.}}
;      YML: --- !Passed
; YML-NEXT: Pass: inline
; YML-NEXT: Name: Inlined
; YML-NEXT: Function: test
; YML-NEXT: Args:
;      YML:  - Callee: f
;      YML:  - Caller: test
;      YML: ...
;  YML-NOT: {{.}}

; BAD-PLUGIN-NOT: {{.}}
;     BAD-PLUGIN: {{.*}}Could not load library {{.*}}nonexistent.so{{.*}}
; BAD-PLUGIN-NOT: {{.}}

;--- host-x86_64-unknown-linux-gnu.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;--- openmp-amdgcn-amd-amdhsa.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @f() {
entry:
  ret void
}

define amdgpu_kernel void @test() {
entry:
  call void @f()
  ret void
}
