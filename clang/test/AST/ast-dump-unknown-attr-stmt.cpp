// An unknown [[...]] attribute is retained on a statement too, not just a
// declaration: UnknownAttr is a DeclOrStmtAttr, so it lands on the AttributedStmt
// the attribute appertains to, with its argument text preserved just like the
// declaration path, and every attribute is kept when several appear on one
// statement. Retaining it never emits an "attribute cannot be applied to a
// statement" diagnostic, because an unrecognized attribute-token is ignored per
// [dcl.attr.grammar]/8. Exercised across standard modes.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -Wno-unknown-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -Wno-unknown-attributes -ast-dump %s | FileCheck %s

void f() {
  // The argument clause is retained verbatim, exactly as on a declaration.
  [[ns::transient(a, b)]] { }
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::transient "(a, b)"
  // CHECK-NEXT:   CompoundStmt

  // Several unknown attributes on one statement are all retained, in order. The
  // retention path does not depend on the statement kind (here an expression
  // statement, not a compound one).
  [[ns::a]] [[ns::b]] 1 + 1;
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::a ""
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::b ""
  // CHECK-NEXT:   BinaryOperator
}
