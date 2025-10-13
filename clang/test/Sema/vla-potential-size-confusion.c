// RUN: %clang_cc1 %s -std=c23 -verify=expected,c -fsyntax-only
// RUN: %clang_cc1 %s -std=c23 -verify=good -fsyntax-only -Wno-vla
// RUN: %clang_cc1 -x c++ %s -verify -fsyntax-only
// RUN: %clang_cc1 -DCARET -fsyntax-only -std=c23 -fno-diagnostics-show-line-numbers -fcaret-diagnostics-max-lines=1 %s 2>&1 | FileCheck %s -strict-whitespace

// good-no-diagnostics

int n, m;      // #decl
int size(int);

void foo(int vla[n], int n); // expected-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                expected-note {{does not refer to this declaration}} \
                                expected-note@#decl {{refers to this declaration instead}}

void bar(int (*vla)[n], int n); // expected-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                   expected-note {{does not refer to this declaration}} \
                                   expected-note@#decl {{refers to this declaration instead}}

void baz(int n, int vla[n]); // no diagnostic expected

void quux(int vla[n + 12], int n); // expected-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                      expected-note {{does not refer to this declaration}} \
                                      expected-note@#decl {{refers to this declaration instead}}

void quibble(int vla[size(n)], int n);  // expected-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                           expected-note {{does not refer to this declaration}} \
                                           expected-note@#decl {{refers to this declaration instead}}

void quobble(int vla[n + m], int n, int m);  // expected-warning 2 {{variable length array size expression refers to declaration from an outer scope}} \
                                                expected-note 2 {{does not refer to this declaration}} \
                                                expected-note@#decl 2 {{refers to this declaration instead}}

// For const int, we still treat the function as having a variably-modified
// type, but only in C.
const int x = 12; // #other-decl
void quorble(int vla[x], int x); // c-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                    c-note {{does not refer to this declaration}} \
                                    c-note@#other-decl {{refers to this declaration instead}}

// For constexpr int, the function has a constant array type. It would be nice
// to diagnose this case as well, but the type system replaces the expression
// with the constant value, and so the information about the name of the
// variable used in the size expression is lost.
constexpr int y = 12;
void quuble(int vla[y], int y); // no diagnostic expected

#ifdef __cplusplus
struct S {
  static int v; // #mem-var
  
  void member_function(int vla[v], int v);  // expected-warning {{variable length array size expression refers to declaration from an outer scope}} \
                                               expected-note {{does not refer to this declaration}} \
                                               expected-note@#mem-var {{refers to this declaration instead}}
};
#endif

#ifdef CARET
// Test that caret locations make sense.
int w;
void quable(int vla[w], int w);

// CHECK: void quable(int vla[w], int w);
// CHECK:                     ^
// CHECK: void quable(int vla[w], int w);
// CHECK:                             ^
// CHECK: int w;
// CHECK:     ^
#endif
