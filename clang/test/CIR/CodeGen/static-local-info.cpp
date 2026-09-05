// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s

// CIRGen attaches the VarDecl facts LoweringPrepare needs (isLocalVarDecl,
// TLSKind, isInline, TemplateSpecializationKind) to static-local guarded
// globals as a #cir.static_local_info attribute, so the facts survive without
// a live ASTContext. This is orthogonal to the #cir.var.decl AST handle, which
// is still attached for consumers that need arbitrary AST properties.

struct HasCtor {
  HasCtor();
  int x;
};

int regular() {
  static HasCtor s;
  return s.x;
}

int tls() {
  static thread_local HasCtor s;
  return s.x;
}

// The thread_local static local materializes a non-default TLS kind, alongside
// the retained AST handle.
// CHECK: @_ZZ3tlsvE1s
// CHECK-SAME: ast = #cir.var.decl.ast
// CHECK-SAME: static_local_info = #cir.static_local_info<local = true, tls = dynamic, is_inline = false, tsk = undeclared>

// CHECK: @_ZZ7regularvE1s
// CHECK-SAME: ast = #cir.var.decl.ast
// CHECK-SAME: static_local_info = #cir.static_local_info<local = true, tls = none, is_inline = false, tsk = undeclared>
