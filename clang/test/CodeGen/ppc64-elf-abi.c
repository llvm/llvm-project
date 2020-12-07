// REQUIRES: powerpc-registered-target

// Verify ABI selection by the front end

// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-ELFv1
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -target-abi elfv1 | FileCheck %s --check-prefix=CHECK-ELFv1
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -target-abi elfv2 | FileCheck %s --check-prefix=CHECK-ELFv2
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-ELFv2
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -target-abi elfv1 | FileCheck %s --check-prefix=CHECK-ELFv1
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -target-abi elfv2 | FileCheck %s --check-prefix=CHECK-ELFv2

// CHECK-ELFv1: define void @func_fab(%struct.fab* noalias sret(%struct.fab) align 4 %agg.result, i64 %x.coerce)
// CHECK-ELFv2: define [2 x float] @func_fab([2 x float] %x.coerce)
struct fab { float a; float b; };
struct fab func_fab(struct fab x) { return x; }
