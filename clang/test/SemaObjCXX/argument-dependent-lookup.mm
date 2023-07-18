// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// <rdar://problem/9142559>: For the purposes of Argument-Dependent
// Lookup, Objective-C classes are considered to be in the global
// namespace.

@interface NSFoo
@end

template<typename T>
void f(T t) {
  g(t);
}

void g(NSFoo*);

void test(NSFoo *foo) {
  f(foo);
}
