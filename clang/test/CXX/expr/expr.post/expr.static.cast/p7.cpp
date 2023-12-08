// RUN: %clang_cc1 -std=c++1z -verify %s -fcxx-exceptions

void (*p)() noexcept;
void (*q)();

void f() {
  // FIXME: This seems like a bad rule.
  p = static_cast<decltype(p)>(q); // expected-error {{not allowed}}
  q = static_cast<decltype(q)>(p);
}
