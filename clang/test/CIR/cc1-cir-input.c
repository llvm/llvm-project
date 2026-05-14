// Verify clang -cc1 accepts a serialized .cir file as input and re-emits it,
// proving the new -x cir entry point round-trips through the frontend.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s \
// RUN:   -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-cir -o - | FileCheck %s --check-prefix=ROUNDTRIP

// Lower a parsed .cir file all the way to LLVM IR. The CIR-to-LLVM dialect
// translation does not require a live ASTContext, so this path is supported
// even before LoweringPrepare is made AST-free.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s \
// RUN:   -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=LLVM

int add(int a, int b) { return a + b; }

// ROUNDTRIP: cir.func {{.*}}@add
// ROUNDTRIP: cir.return

// LLVM: define {{.*}} i32 @add(i32 {{.*}}%[[A:.+]], i32 {{.*}}%[[B:.+]])
// LLVM: ret i32
