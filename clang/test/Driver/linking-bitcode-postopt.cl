// REQUIRES: amdgpu-registered-target

// Test that -mlink-bitcode-postopt correctly enables LinkInModulesPass

// RUN: %clang -c -mllvm -print-pipeline-passes -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefixes=DEFAULT %s

// DEFAULT-NOT: LinkInModulesPass

// RUN: %clang -c -mllvm -print-pipeline-passes -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   -Xclang -mlink-builtin-bitcode-postopt \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefixes=OPTION-POSITIVE %s

// OPTION-POSITIVE: LinkInModulesPass

// RUN: %clang -c -mllvm -print-pipeline-passes -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   -Xclang -mno-link-builtin-bitcode-postopt \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefixes=OPTION-NEGATIVE %s

// OPTION-NEGATIVE-NOT: LinkInModulesPass

// RUN: %clang -c -mllvm -print-pipeline-passes -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   -Xclang -mlink-builtin-bitcode-postopt \
// RUN:   -Xclang -mno-link-builtin-bitcode-postopt \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefixes=OPTION-POSITIVE-NEGATIVE %s

// OPTION-POSITIVE-NEGATIVE-NOT: LinkInModulesPass

kernel void func(void);
