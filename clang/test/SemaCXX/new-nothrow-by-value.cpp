// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

// [expr.const] only permits a new-expression in a constant expression when it
// selects a replaceable global allocation function. None of them take
// std::nothrow_t by value, so a user-declared allocation function that does
// must be rejected before examining its placement argument. The constant
// evaluator used to assume the argument was always an lvalue and crashed on
// the prvalue produced here.

namespace std {
inline constexpr struct nothrow_t {
} nothrow;
} // namespace std

void *operator new[](__SIZE_TYPE__, std::nothrow_t) noexcept;
void *operator new(__SIZE_TYPE__, std::nothrow_t) noexcept;

void set(int *p) {
  p = (1 ? new (std::nothrow) int[1] : nullptr);
  p = (1 ? new (std::nothrow) int : nullptr);
}

constexpr bool by_value() { // expected-error {{constexpr function never produces a constant expression}}
  int *p = new (std::nothrow) int; // expected-note 2{{call to placement 'operator new'}}
  delete p;
  return true;
}
static_assert(by_value()); // expected-error {{not an integral constant expression}} expected-note {{in call to 'by_value()'}}
