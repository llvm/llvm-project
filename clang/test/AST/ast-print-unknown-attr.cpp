// An unknown C++ [[...]] attribute, retained as an UnknownAttr, prints back to
// equivalent source under -ast-print: the scope, name and the source text of
// the argument clause are reproduced. This is the round-trip oracle for the
// retention feature.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-print %s | FileCheck %s

struct X {
  int x [[ns::transient(a, b)]];
};
// CHECK: struct X {
// CHECK-NEXT: int x {{\[\[}}ns::transient(a, b){{\]\]}};
// CHECK-NEXT: };

[[frobble]] void g();
// CHECK: {{\[\[}}frobble{{\]\]}} void g();

[[ns::plain]] void h();
// CHECK: {{\[\[}}ns::plain{{\]\]}} void h();

// The argument clause is retained as the verbatim source span between its
// parentheses, so a macro used as an argument stays unexpanded and comments are
// kept.
#define M 1 + 2
[[vendor::attr(M)]] int a;
// CHECK: {{\[\[}}vendor::attr(M){{\]\]}} int a;

[[vendor::attr(1 /*c*/ + 2)]] int b;
// CHECK: {{\[\[}}vendor::attr(1 /*c*/ + 2){{\]\]}} int b;

// When the parentheses themselves come from a macro expansion, the clause maps
// to no file range, so the argument text is dropped; the attribute is still
// retained.
#define WHOLE [[vendor::attr(9)]]
WHOLE int c;
// CHECK: {{\[\[}}vendor::attr{{\]\]}} int c;
