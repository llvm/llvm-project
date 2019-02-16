// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -ast-dump -ast-dump-filter AutoVar | FileCheck %s

namespace {
  class foo {
  };
}

#pragma GCC visibility push(hidden)
auto AutoVar = foo();

// CHECK: VarDecl {{.*}} AutoVar
// CHECK-NOT: VisibilityAttr
