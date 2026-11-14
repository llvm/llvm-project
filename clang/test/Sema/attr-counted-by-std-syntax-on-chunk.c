// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -fexperimental-late-parse-attributes -verify %s
// RUN: not %clang_cc1 -std=c23 -fsyntax-only -ast-dump %s 2>/dev/null | FileCheck %s
// RUN: not %clang_cc1 -std=c23 -fsyntax-only -fexperimental-late-parse-attributes -ast-dump %s 2>/dev/null | FileCheck %s

// A [[]] counted_by-family attribute on a declarator chunk is a type
// attribute. It applies to a field's pointer, and is rejected rather than
// dropped elsewhere. It isn't late-parsed, so the late-parsing mode handles it
// the same way.

void f(int n, int *[[clang::counted_by(n)]] p); // expected-error{{'clang::counted_by' attribute cannot be applied to types}}

int gn;
int *[[clang::sized_by(gn)]] gp; // expected-error{{'clang::sized_by' attribute cannot be applied to types}}

struct S {
  int n;
  // CHECK: FieldDecl {{.*}} p 'int * __counted_by_or_null(n)':'int *'
  int *[[clang::counted_by_or_null(n)]] p;
  int *[[clang::counted_by(n)]] *pp; // expected-error{{'counted_by' attribute on nested pointer type is not allowed}}
};
