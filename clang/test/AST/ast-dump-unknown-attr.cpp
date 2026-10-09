// An otherwise-unknown C++ [[...]] attribute is retained in the AST as an
// UnknownAttr instead of being dropped after the -Wunknown-attributes
// diagnostic, so tooling and plugins can recover it.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-dump %s \
// RUN:   | FileCheck %s

struct X {
  int x [[ns::transient(a, b)]];
};

// CHECK: FieldDecl {{.*}} x 'int'
// The dump identifies the retained attribute by its scope::name.
// CHECK: UnknownAttr {{.*}} ns::transient
