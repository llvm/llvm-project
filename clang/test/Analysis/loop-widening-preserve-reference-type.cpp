// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-max-loop 4 -analyzer-config widen-loops=true -verify %s

void clang_analyzer_eval(int);

struct A {
  ~A() {}
};
struct B : public A {};

void invalid_type_region_access() {
  const A &x = B();
  for (int i = 0; i < 10; ++i) { }
  clang_analyzer_eval(&x != 0); // expected-warning{{TRUE}}
}                               // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

using AR = const A &;
void invalid_type_alias_region_access() {
  AR x = B();
  for (int i = 0; i < 10; ++i) {
  }
  clang_analyzer_eval(&x != 0); // expected-warning{{TRUE}}
} // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
