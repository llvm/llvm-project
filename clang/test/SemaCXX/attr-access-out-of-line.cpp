// RUN: %clang_cc1 -std=c++20 -Wthread-safety -fsyntax-only -verify %s

struct __attribute__((capability("mutex"))) Mutex {};

namespace ns {
class C {
private:
  Mutex mu_;                        // expected-note 2 {{declared private here}}
  static constexpr int kAlign = 16; // expected-note 5 {{declared private here}}

public:
  C() __attribute__((locks_excluded(mu_)));
  [[clang::requires_capability(mu_)]] ~C();
  void f1() __attribute__((locks_excluded(mu_)));
  [[clang::requires_capability(mu_)]] void f2();
  void f3() __attribute__((locks_excluded(mu_)));
  void f4() __attribute__((locks_excluded(mu_)));
  static int s1;
  static int s2;
  static int s3;
  friend void friend_fn();
};

// Valid out-of-line definitions in namespace scope:
__attribute__((locks_excluded(C::mu_))) void C::f1() {}

[[clang::requires_capability(C::mu_)]] void C::f2() {}

__attribute__((unused)) [[clang::requires_capability(C::mu_)]]
__attribute__((locks_excluded(C::mu_))) [[maybe_unused]]
void C::f3() {}

__attribute__((locks_excluded(C::mu_))) inline void C::f4() {}

__attribute__((locks_excluded(C::mu_))) C::C() {}
[[clang::requires_capability(C::mu_)]] C::~C() {}

__attribute__((aligned(C::kAlign))) int C::s1 = 0;
[[gnu::aligned(C::kAlign)]] int C::s2 = 0;
static_assert(alignof(decltype(C::s1)) <= alignof(int));

__attribute__((locks_excluded(C::mu_))) void friend_fn() {}

// Invalid private accesses on unrelated declarations must still be diagnosed:
__attribute__((locks_excluded(C::mu_))) // expected-error {{'mu_' is a private member of 'ns::C'}}
void unrelated_fn() {}

[[clang::requires_capability(C::mu_)]] // expected-error {{'mu_' is a private member of 'ns::C'}}
void unrelated_fn2() {}

// Multi-declarator group: C::s3 is allowed, unrelated_var is diagnosed.
__attribute__((aligned(C::kAlign))) int C::s3 = 0, unrelated_var = 0; // expected-error {{'kAlign' is a private member of 'ns::C'}}

// Non-declarator / non-DeclOrFuncDef constructs must not drop diagnostics:
__attribute__((aligned(C::kAlign))) typedef int MyInt; // expected-error {{'kAlign' is a private member of 'ns::C'}}
__attribute__((aligned(C::kAlign))) using MyInt2 = int; // expected-error {{'kAlign' is a private member of 'ns::C'}} \
                                                        // expected-error {{misplaced attributes}}
__attribute__((aligned(C::kAlign))); // expected-error {{'kAlign' is a private member of 'ns::C'}}
static_assert(C::kAlign == 16); // expected-error {{'kAlign' is a private member of 'ns::C'}}
} // namespace ns

// Global scope & extern "C++" { ... }:
class GlobalC {
private:
  Mutex mu_; // expected-note {{declared private here}}

public:
  void g1() __attribute__((locks_excluded(mu_)));
  [[clang::requires_capability(mu_)]] void g2();
  void g3() __attribute__((locks_excluded(mu_)));
};

__attribute__((locks_excluded(GlobalC::mu_))) void GlobalC::g1() {}

[[clang::requires_capability(GlobalC::mu_)]] void GlobalC::g2() {}

extern "C++" {
__attribute__((locks_excluded(GlobalC::mu_))) void GlobalC::g3() {}

__attribute__((locks_excluded(GlobalC::mu_))) // expected-error {{'mu_' is a private member of 'GlobalC'}}
void global_unrelated() {}
}
