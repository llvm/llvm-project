// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wc++11-compat
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
  int auto b; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
  auto c; // expected-warning {{C++11 extension}} expected-error {{requires an initializer}}
  static auto d = 0; // expected-warning {{C++11 extension}}
  auto static e = 0; // expected-warning {{C++11 extension}}
}

// typedef and auto storage-class-specifier cannot appear in the same
// decl-specifier-seq ([dcl.stc] p1). This must be diagnosed in C++98 even
// though 'auto int' (without typedef) is valid there.
void g() {
  typedef auto int t1;    // expected-error {{cannot combine with previous 'typedef' declaration specifier}}
  auto typedef int t2;    // expected-error {{cannot combine with previous 'typedef' declaration specifier}}
}
