; Tests the clang-sycl-linker tool: device code splitting.
;
; REQUIRES: spirv-registered-target
;
; RUN: llvm-as %s -o %t.bc
;
; Test that an invalid split mode is rejected.
; RUN: not clang-sycl-linker --dry-run -triple=spirv64 --module-split-mode=bogus %t.bc -o a.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-INVALID
; SPLIT-INVALID: module-split-mode value isn't recognized: bogus
;
; Test the split mode ("none"): no extra splits are produced.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t.bc -o %t-none.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-NONE
; SPLIT-NONE:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SPLIT-NONE-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
; SPLIT-NONE-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
; SPLIT-NONE-NOT:  LLVM backend: input: {{.*}}.bc, output: {{.*}}_1.spv
;
; Test per-kernel split: a module with two SPIR_KERNEL functions produces two
; device images.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=kernel %t.bc -o %t-split-kernel.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-KERNEL
; SPLIT-KERNEL:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SPLIT-KERNEL-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[SPLIT0:.*]].bc, [[SPLIT1:.*]].bc, mode: kernel
; SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT0]].bc, output: {{.*}}_0.spv
; SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT1]].bc, output: {{.*}}_1.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper_shared(i32 %a) {
entry:
  %r = add nsw i32 %a, 1
  ret i32 %r
}

define spir_kernel void @kernel_a(ptr addrspace(1) %out, i32 %a) {
entry:
  %r = tail call spir_func i32 @helper_shared(i32 %a)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @kernel_b(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %h = tail call spir_func i32 @helper_shared(i32 %a)
  %r = mul nsw i32 %h, %b
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}
