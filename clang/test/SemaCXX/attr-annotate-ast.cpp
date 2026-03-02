// RUN: %clang_cc1 -std=gnu++20 -fsyntax-only -ast-dump %s | FileCheck %s

void f() {
  [[clang::annotate("decl", 1)]] int i = 0;
  [[clang::annotate("stmt", 2)]] i += 1;
[[clang::annotate("label", 3)]] label1:
  i += 2;
}

// CHECK: -FunctionDecl {{.*}} f 'void ()'
// CHECK: -VarDecl {{.*}} used i 'int'
// CHECK: -AnnotateAttr {{.*}} "decl"
// CHECK: -IntegerLiteral {{.*}} 'int' 1
// CHECK: -AttributedStmt
// CHECK: -AnnotateAttr {{.*}} "stmt"
// CHECK: -IntegerLiteral {{.*}} 'int' 2
// CHECK: -LabelStmt {{.*}} 'label1'
// CHECK: -AnnotateAttr {{.*}} "label"
// CHECK: -IntegerLiteral {{.*}} 'int' 3
// CHECK: -CompoundAssignOperator

template <typename T> void g() {
  [[clang::annotate("tmpl_decl", 4)]] T j = 0;
  [[clang::annotate("tmpl_stmt", 5)]] j += 1;
[[clang::annotate("tmpl_label", 6)]] label2:
  j += 2;
}

// CHECK: -FunctionTemplateDecl {{.*}} g
// CHECK: -VarDecl {{.*}} referenced j 'T'
// CHECK: -AnnotateAttr {{.*}} "tmpl_decl"
// CHECK: -IntegerLiteral {{.*}} 'int' 4
// CHECK: -AttributedStmt
// CHECK: -AnnotateAttr {{.*}} "tmpl_stmt"
// CHECK: -IntegerLiteral {{.*}} 'int' 5
// CHECK: -LabelStmt {{.*}} 'label2'
// CHECK: -AnnotateAttr {{.*}} "tmpl_label"
// CHECK: -IntegerLiteral {{.*}} 'int' 6
// CHECK: -CompoundAssignOperator

void h() {
  g<int>();
}

// CHECK: -FunctionDecl {{.*}} used g 'void ()' implicit_instantiation
// CHECK: -VarDecl {{.*}} used j 'int'
// CHECK: -AnnotateAttr {{.*}} "tmpl_decl"
// CHECK: -IntegerLiteral {{.*}} 'int' 4
// CHECK: -AttributedStmt
// CHECK: -AnnotateAttr {{.*}} Implicit "tmpl_stmt"
// CHECK: -IntegerLiteral {{.*}} 'int' 5
// CHECK: -LabelStmt {{.*}} 'label2'
// CHECK: -AnnotateAttr {{.*}} "tmpl_label"
// CHECK: -IntegerLiteral {{.*}} 'int' 6
// CHECK: -CompoundAssignOperator
