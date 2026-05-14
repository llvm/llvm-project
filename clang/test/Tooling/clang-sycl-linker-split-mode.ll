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
; Test the split mode ("none"): kernels from different TUs are not split into separate images.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t.bc -o %t-none.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-NONE
; SPLIT-NONE:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SPLIT-NONE-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
; SPLIT-NONE-NOT:  {{.+}}
;
; Test the split mode ("kernel"): each SPIR_KERNEL function produces its own device image.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=kernel %t.bc -o %t-split-kernel.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-KERNEL
; SPLIT-KERNEL:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SPLIT-KERNEL-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, mode: kernel
; SPLIT-KERNEL-NEXT: [[SPLIT0:.*]].bc [kernel_c ]
; SPLIT-KERNEL-NEXT: [[SPLIT1:.*]].bc [kernel_b ]
; SPLIT-KERNEL-NEXT: [[SPLIT2:.*]].bc [kernel_a ]
; SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT0]].bc, output: {{.*}}_0.spv
; SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT1]].bc, output: {{.*}}_1.spv
; SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT2]].bc, output: {{.*}}_2.spv
; SPLIT-KERNEL-NOT:  {{.+}}
;
; Test default split mode ('source'): no --module-split-mode flag needed.
; Two kernels with different sycl-module-id values produce two device images.
; sycl_external function is not treated as entry point and doesn't produce a separate image
; despite having a different sycl-module-id.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t.bc -o %t-src.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-SRC
;
; Test per-TU split ('source' explicitly provided)
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=source %t.bc -o %t-src.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SPLIT-SRC
; SPLIT-SRC:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SPLIT-SRC-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, mode: source
; SPLIT-SRC-NEXT: [[S0:.*]].bc [kernel_b kernel_c ]
; SPLIT-SRC-NEXT: [[S1:.*]].bc [kernel_a ]
; SPLIT-SRC-NEXT: LLVM backend: input: [[S0]].bc, output: {{.*}}_0.spv
; SPLIT-SRC-NEXT: LLVM backend: input: [[S1]].bc, output: {{.*}}_1.spv
; SPLIT-SRC-NOT:  {{.+}}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper() {
  ret i32 0
}

define spir_kernel void @kernel_a() #0 {
  %r = call spir_func i32 @helper()
  ret void
}

define spir_kernel void @kernel_b() #1 {
  %r = call spir_func i32 @helper()
  ret void
}

define spir_kernel void @kernel_c() #1 {
  %r = call spir_func i32 @helper()
  ret void
}

define spir_func i32 @ext_fn() #2 {
  %r = call spir_func i32 @helper()
  ret i32 0
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
attributes #2 = { "sycl-module-id"="TU3.cpp" }
