// REQUIRES: amdgpu-registered-target

// Test that -mlink-bitcode-postopt correctly enables LinkInModulesPass

// RUN: %clang_cc1 -triple amdgcn-- -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN: %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s

// DEFAULT-NOT: LinkInModulesPass

// RUN: %clang_cc1 -triple amdgcn-- -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mlink-builtin-bitcode-postopt \
// RUN: %s 2>&1 | FileCheck --check-prefixes=OPTION-POSITIVE %s

// OPTION-POSITIVE: LinkInModulesPass

// RUN: %clang_cc1 -triple amdgcn-- -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mno-link-builtin-bitcode-postopt \
// RUN: %s 2>&1 | FileCheck --check-prefixes=OPTION-NEGATIVE %s

// OPTION-NEGATIVE-NOT: LinkInModulesPass

// RUN: %clang_cc1 -triple amdgcn-- -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mlink-builtin-bitcode-postopt \
// RUN:   -mno-link-builtin-bitcode-postopt \
// RUN: %s 2>&1 | FileCheck --check-prefixes=OPTION-POSITIVE-NEGATIVE %s

// OPTION-POSITIVE-NEGATIVE-NOT: LinkInModulesPass
