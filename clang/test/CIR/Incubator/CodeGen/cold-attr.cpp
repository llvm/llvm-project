// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

// Test: Function with cold attribute
// CIR: cir.func{{.*}}cold{{.*}}@_Z9cold_funcv
__attribute__((cold))
void cold_func() {}

// LLVM-LABEL: define{{.*}}@_Z9cold_funcv
// LLVM-SAME: [[COLD_ATTRS:#[0-9]+]]
// LLVM: attributes [[COLD_ATTRS]] = {{{.*}}cold{{.*}}}

// OGCG-LABEL: define{{.*}}@_Z9cold_funcv
// OGCG-SAME: [[COLD_ATTRS_OG:#[0-9]+]]
// OGCG: attributes [[COLD_ATTRS_OG]] = {{{.*}}cold{{.*}}}
