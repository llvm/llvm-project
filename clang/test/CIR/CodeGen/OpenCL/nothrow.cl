// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-cir -o %t.cir %s
// RUN: FileCheck %s -input-file=%t.cir -check-prefixes CIR
// RUN: %clang_cc1 -fclangir -triple=spirv64-unknown-unknown -emit-llvm -o %t.ll %s
// RUN: FileCheck %s -input-file=%t.ll -check-prefixes LLVM

// CIR-LABEL: #fn_attr =
// CIR: cl.kernel = #cir.cl.kernel
// CIR: nothrow = #cir.nothrow

// CIR-LABEL: #fn_attr1 =
// CIR-NOT: cl.kernel = #cir.cl.kernel
// CIR: nothrow = #cir.nothrow

kernel void ker() {};
// CIR: cir.func @ker{{.*}} extra(#fn_attr) {
// LLVM: define{{.*}}@ker(){{.*}} #0

void foo() {};
// CIR: cir.func @foo{{.*}} extra(#fn_attr1) {
// LLVM: define{{.*}}@foo(){{.*}} #1

// LLVM-LABEL: attributes #0
// LLVM: nounwind

// LLVM-LABEL: attributes #1
// LLVM: nounwind
