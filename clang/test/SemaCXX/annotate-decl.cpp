// RUN: %clang_cc1 %s -std=c++17 -fsyntax-only -fcxx-exceptions -verify

[[clang::annotate_decl("foo")]] int global1;
int [[clang::annotate_decl("foo")]] global2; // expected-error {{'annotate_decl' attribute cannot be applied to types}}

// Redeclarations.

// A declaration that originally didn't have an `annotate_decl` attribute
// can be redeclared with one.
extern int global3;
[[clang::annotate_decl("foo")]] extern int global3;

// A declaration that originally had an `annotate_decl` attribute can be
// redeclared without one.
[[clang::annotate_decl("foo")]] extern int global4;
extern int global4;

// A declaration that originally had an `annotate_decl` attribute with one
// set of arguments can be redeclared with another set of arguments.
[[clang::annotate_decl("foo", "arg1", 1)]] extern int global5;
[[clang::annotate_decl("foo", "arg2", 2)]] extern int global5;

// Different types of declarations.
[[clang::annotate_decl("foo")]] void f();
namespace [[clang::annotate_decl("foo")]] my_namespace {}
struct [[clang::annotate_decl("foo")]] S;
struct [[clang::annotate_decl("foo")]] S{
  [[clang::annotate_decl("foo")]] int member;
};
template <class T>
[[clang::annotate_decl("foo")]] T var_template;
extern "C" [[clang::annotate_decl("foo")]] int extern_c_func();
[[clang::annotate_decl("foo")]] extern "C" int extern_c_func(); // expected-error {{an attribute list cannot appear here}}

// Declarations within functions.
void f2() {
  [[clang::annotate_decl("foo")]] int i;
  [[clang::annotate_decl("foo")]] i = 1; // expected-error {{'annotate_decl' attribute cannot be applied to a statement}}

  // Test various cases where a declaration can appear inside a statement.
  for ([[clang::annotate_decl("foo")]] int i = 0; i < 42; ++i) {}
  for (; [[clang::annotate_decl("foo")]] bool b = false;) {}
  while ([[clang::annotate_decl("foo")]] bool b = false) {}
  if ([[clang::annotate_decl("foo")]] bool b = false) {}
  try {
  } catch ([[clang::annotate_decl("foo")]] int i) {
  }
}
