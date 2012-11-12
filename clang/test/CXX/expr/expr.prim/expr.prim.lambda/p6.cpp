// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

void test_conversion() {
  int (*fp1)(int) = [](int x) { return x + 1; };
  void (*fp2)(int) = [](int x) { };

  const auto lambda = [](int x) { };
  void (*fp3)(int) = lambda;

  volatile const auto lambda2 = [](int x) { }; // expected-note{{but method is not marked volatile}}
  void (*fp4)(int) = lambda2; // expected-error{{no viable conversion}}
}

void test_no_conversion() { 
  int (*fp1)(int) = [=](int x) { return x + 1; }; // expected-error{{no viable conversion}}
  void (*fp2)(int) = [&](int x) { }; // expected-error{{no viable conversion}}
}

void test_wonky() {
  const auto l = [](int x) mutable -> int { return + 1; };
  l(17); // okay: uses conversion function
}
