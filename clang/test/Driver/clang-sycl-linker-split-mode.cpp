// Tests the clang-sycl-linker tool: device code splitting.
//
// REQUIRES: spirv-registered-target
//
// Test that an invalid split mode is rejected.
// RUN: llvm-as %S/Inputs/SYCL/multimodules.ll -o %t_multimodules.bc
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 --module-split-mode=bogus %t_multimodules.bc -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-INVALID
// SPLIT-INVALID: module-split-mode value isn't recognized: bogus
//
// Test the split mode ("none"): kernels from different TUs are not split into separate images.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t_multimodules.bc -o %t-split-none.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-NONE
// SPLIT-NONE:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SPLIT-NONE-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
// SPLIT-NONE-NOT:  {{.+}}
//
// Test the split mode ("kernel"): a module with two SPIR_KERNEL functions produces two device images.
// RUN: llvm-as %S/Inputs/SYCL/two-kernels.ll -o %t-two.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=kernel %t-two.bc -o %t-split-kernel.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-KERNEL
// SPLIT-KERNEL:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SPLIT-KERNEL-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[SPLIT0:.*]].bc, [[SPLIT1:.*]].bc, mode: kernel
// SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT0]].bc, output: {{.*}}_0.spv
// SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT1]].bc, output: {{.*}}_1.spv
//
// Test default split mode ('source'): no --module-split-mode flag needed.
// Two kernels with different sycl-module-id values produce two device images.
// sycl_external function is not treated as entry point and doesn't produce a separate image
// despite having a different sycl-module-id.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_multimodules.bc -o %t-src.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-SRC
//
// Test per-TU split ('source' explicitly provided)
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=source %t_multimodules.bc -o %t-src.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-SRC
// SPLIT-SRC:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SPLIT-SRC-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[S0:.*]].bc, [[S1:.*]].bc, mode: source
// SPLIT-SRC-NEXT: LLVM backend: input: [[S0]].bc, output: {{.*}}_0.spv
// SPLIT-SRC-NEXT: LLVM backend: input: [[S1]].bc, output: {{.*}}_1.spv
// SPLIT-SRC-NOT:  {{.+}}
