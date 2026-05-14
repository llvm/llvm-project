// RUN: %clang_cc1 -fsyntax-only -std=c++17 -ferror-limit 19 -verify %s

// After -ferror-limit is exceeded the parser enters a fatal-error state.
// Late-parsed attribute conditions must not be attached to the declaration
// in that state: name lookup of the condition's sub-expressions may fail
// silently and wrap the result in a RecoveryExpr, producing a
// value-dependent condition on a non-template function that later confuses
// the diagnose_if consumer (Expr::EvaluateWithSubstitution asserts on
// value-dependent expressions).
// We test that we do not crash in such cases (#197625).

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

using size_t = decltype(sizeof(0));
namespace std {
template <typename T>
struct initializer_list {
  const T *p;
  size_t s;
  constexpr size_t size() const { return s; }
};
} // namespace std

template <typename T>
struct E {
  void f(int i) _diagnose_if(i, "bad i", "error"); // expected-note 19 {{from 'diagnose_if' attribute on 'f'}}
};
void blast() {
  E<int> e;
  e.f(1); e.f(1); e.f(1); e.f(1); e.f(1); // expected-error 5 {{bad i}}
  e.f(1); e.f(1); e.f(1); e.f(1); e.f(1); // expected-error 5 {{bad i}}
  e.f(1); e.f(1); e.f(1); e.f(1); e.f(1); // expected-error 5 {{bad i}}
  e.f(1); e.f(1); e.f(1); e.f(1); e.f(1); // expected-error 4 {{bad i}} expected-error@* {{too many errors emitted}}
}

struct Foo {
  Foo(std::initializer_list<int> l)
    _diagnose_if(l.size() == 1, "first", "warning")
    _diagnose_if(l.size() == 2, "second", "error") {}
};

void run() {
  Foo{std::initializer_list<int>{}};
  Foo{std::initializer_list<int>{1}};
  Foo{std::initializer_list<int>{1, 2}};
}
