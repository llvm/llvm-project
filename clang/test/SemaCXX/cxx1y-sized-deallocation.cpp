// RUN: %clang_cc1 -std=c++1y -verify %s -fsized-deallocation -fexceptions -fcxx-exceptions

using size_t = decltype(sizeof(0));
void operator delete(void *, size_t) noexcept;   // expected-note {{'operator delete' declared here}}
void operator delete[](void *, size_t) noexcept;

void f(void *p, void *q) {
  // OK, implicitly declared.
  operator delete(p, 8);
  operator delete[](q, 12);
  static_assert(noexcept(operator delete(p, 8)), "");
  static_assert(noexcept(operator delete[](q, 12)), "");
}

void *operator new(size_t bad, size_t idea);
struct S { S() { throw 0; } };
void g() {
  new (123) S; // expected-error {{'new' expression with placement arguments refers to non-placement 'operator delete'}}
}
