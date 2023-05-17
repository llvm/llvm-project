// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s

// Similiar to declarator-function.cpp, but for member functions.
class Foo {
  void foo() {};
// CHECK-NOT: member-declarator := declarator brace-or-equal-initializer
// CHECK: member-declaration~function-definition := decl-specifier-seq function-declarator function-body
// CHECK-NOT: member-declarator := declarator brace-or-equal-initializer
};
