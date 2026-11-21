// REQUIRES: spirv-registered-target

// Intel IGCA targets provided via -target-cpu should not result in diagnostics
// from the SPIR-V LLVM backend.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -target-cpu igca_20ca -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGCA
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -target-cpu igca_20ca -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGCA
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE

int f(int x) { return x + 1; }

// IGCA-NOT: not a recognized processor
// IGCA: "target-cpu"="igca_20ca"

// NONE-NOT: not a recognized processor
// NONE-NOT: "target-cpu"
