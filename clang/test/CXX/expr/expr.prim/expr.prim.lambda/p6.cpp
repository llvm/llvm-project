// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++1z %s -verify

void test_conversion() {
  int (*fp1)(int) = [](int x) { return x + 1; };
  void (*fp2)(int) = [](int x) { };

  const auto lambda = [](int x) { };
  void (*fp3)(int) = lambda;

  volatile const auto lambda2 = [](int x) { }; // expected-note{{but method is not marked volatile}}
  void (*fp4)(int) = lambda2; // expected-error{{no viable conversion}}

  void (*fp5)(int) noexcept = [](int x) { };
#if __cplusplus > 201402L
  // expected-error@-2 {{no viable}} expected-note@-2 {{candidate}}
  void (*fp5a)(int) noexcept = [](auto x) { };
  // expected-error@-1 {{no viable}} expected-note@-1 {{candidate}}
  void (*fp5b)(int) noexcept = [](auto x) noexcept { };
#endif
  void (*fp6)(int) noexcept = [](int x) noexcept { };
}

void test_no_conversion() { 
  int (*fp1)(int) = [=](int x) { return x + 1; }; // expected-error{{no viable conversion}}
  void (*fp2)(int) = [&](int x) { }; // expected-error{{no viable conversion}}
}

void test_wonky() {
  const auto l = [](int x) mutable -> int { return + 1; };
  l(17); // okay: uses conversion function
}
