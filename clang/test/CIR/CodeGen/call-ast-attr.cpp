// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir -mlir-print-op-generic %s -o - | FileCheck %s

struct S {
  void method();
};

// Member calls carry the AST call expression as well.
void member_call(S &s) { s.method(); }
// CHECK: "cir.call"
// CHECK-SAME: ast = #cir.call.expr.ast

// Calls through pointers to member functions carry it too.
void member_ptr_call(S &s, void (S::*fn)()) { (s.*fn)(); }
// CHECK: "cir.call"
// CHECK-SAME: ast = #cir.call.expr.ast
