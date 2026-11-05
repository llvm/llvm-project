// Offload arch names accepted as -target-cpu on SPIR-V must not be forwarded to
// the LLVM backend, which defines no processors and would print a
// non-diagnostic "not a recognized processor" message to stderr. The
// "target-cpu" IR attribute is still emitted.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -target-cpu igca_20ca -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IGCA
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -target-cpu xe-pvc -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=XE
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE

int f(int x) { return x + 1; }

// IGCA-NOT: not a recognized processor
// IGCA: "target-cpu"="igca_20ca"
// IGCA-NOT: not a recognized processor

// XE-NOT: not a recognized processor
// XE: "target-cpu"="xe-pvc"
// XE-NOT: not a recognized processor

// NONE-NOT: not a recognized processor
// NONE-NOT: "target-cpu"
