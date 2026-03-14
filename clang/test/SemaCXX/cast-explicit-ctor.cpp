// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
struct B { explicit B(bool); };
void f() { 
  (void)(B)true;
  (void)B(true); 
}
