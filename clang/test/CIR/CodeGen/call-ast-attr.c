// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir -mlir-print-op-generic %s -o - | FileCheck %s --check-prefix=GENERIC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=DEFAULT

void callee(void);

void caller(void) { callee(); }

// Calls carry the AST call expression, visible in the generic form.
// GENERIC: "cir.call"(){{.*}}ast = #cir.call.expr.ast

// The attribute is elided from the pretty form.
// DEFAULT: cir.call @callee()
// DEFAULT-NOT: #cir.call.expr.ast
