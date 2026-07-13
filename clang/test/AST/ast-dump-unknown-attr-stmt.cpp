// An unknown [[...]] attribute is retained on a statement too, not just a
// declaration: UnknownAttr is a DeclOrStmtAttr, so it lands on the AttributedStmt
// the attribute appertains to. Retaining it never emits an "attribute cannot be
// applied to a statement" diagnostic, because an unrecognized attribute-token is
// ignored per [dcl.attr.grammar]/8. Retention is consistent across statement
// kinds, carries the argument text just like the declaration path, and keeps
// every attribute when several appear on one statement. Exercised across
// standard modes.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -Wno-unknown-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -Wno-unknown-attributes -ast-dump %s | FileCheck %s

void f() {
  // On a compound statement, with an argument clause retained verbatim, exactly
  // as on a declaration.
  [[ns::transient(a, b)]] { }
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::transient "(a, b)"
  // CHECK-NEXT:   CompoundStmt

  // On a loop statement.
  [[ns::loop]] for (;;) { break; }
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::loop ""
  // CHECK-NEXT:   ForStmt

  // On an expression statement.
  [[vendor::note]] 1 + 1;
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} vendor::note ""
  // CHECK-NEXT:   BinaryOperator

  // On a null statement.
  [[vendor::skip]];
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} vendor::skip ""
  // CHECK-NEXT:   NullStmt

  // Several unknown attributes on one statement are all retained, in order.
  [[ns::a]] [[ns::b]] { }
  // CHECK:      AttributedStmt
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::a ""
  // CHECK-NEXT:   UnknownAttr {{.*}} ns::b ""
  // CHECK-NEXT:   CompoundStmt
}
