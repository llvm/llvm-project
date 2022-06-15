// RUN: %clang_cc1 %s -ast-dump -fdouble-square-bracket-attributes | FileCheck %s

// Verify that we print the [[clang::annotate_type]] attribute.
// FIXME: The arguments are currently not printed -- see also comments in
// TypePrinter.cpp.

// Need to escape the `[[` as a regex to avoid it being interpreted as a
// substitution block.
// CHECK: VarDecl {{.*}} x1 'int {{\[\[}}clang::annotate_type(...){{]]}}':'int'
int [[clang::annotate_type("bar")]] x1;
// CHECK: VarDecl {{.*}} x2 'int * {{\[\[}}clang::annotate_type(...){{]]}}':'int *'
int *[[clang::annotate_type("bar")]] x2;
