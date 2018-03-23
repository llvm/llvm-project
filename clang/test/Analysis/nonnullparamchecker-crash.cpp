// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
class C {};

// expected-no-diagnostics
void f(C i) {
  auto lambda = [&] { f(i); };
  typedef decltype(lambda) T;
  T* blah = new T(lambda);
  (*blah)();
  delete blah;
}
