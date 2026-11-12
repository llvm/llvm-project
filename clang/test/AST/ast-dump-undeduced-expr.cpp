// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck %s

struct Foo {
  static constexpr auto Bar = ;
};

// CHECK: -VarDecl {{.*}} invalid Bar 'const auto' static {{(inline )?}}constexpr
