// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Make sure this doesn't crash.

@protocol P
@end
template <class T>
id<P> foo(T) {
  int i;
  foo(i);
}
