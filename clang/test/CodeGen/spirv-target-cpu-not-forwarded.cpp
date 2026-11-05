// SPIRV offload arch names accepted via -target-cpu should not be forwarded
// to LLVM backend: The SPIRV backend only defines a "generic" processor, and
// any other name forwarded results in a "not a recognized processor"
// diagnostic. -target-cpu values are only used in the frontend for SPIRV.
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
