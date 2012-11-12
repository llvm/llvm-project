// RUN: %clang_cc1 -std=c++11 %s -Winvalid-noreturn -verify

// An attribute-specifier-seq in a lambda-declarator appertains to the
// type of the corresponding function call operator.
void test_attributes() {
  auto nrl = [](int x) -> int { if (x > 0) return x; }; // expected-warning{{control may reach end of non-void lambda}}

  auto nrl2 = []() [[noreturn]] { return; }; // expected-error{{lambda declared 'noreturn' should not return}}
}

template<typename T>
struct bogus_override_if_virtual : public T {
  bogus_override_if_virtual() : T(*(T*)0) { }
  int operator()() const;
};

void test_quals() {
  // This function call operator is declared const (9.3.1) if and only
  // if the lambda- expression's parameter-declaration-clause is not
  // followed by mutable.
  auto l = [=](){}; // expected-note{{method is not marked volatile}}
  const decltype(l) lc = l;
  l();
  lc();

  auto ml = [=]() mutable{}; // expected-note{{method is not marked const}} \
                             // expected-note{{method is not marked volatile}} 
  const decltype(ml) mlc = ml;
  ml();
  mlc(); // expected-error{{no matching function for call to object of type}}

  // It is neither virtual nor declared volatile.
  volatile decltype(l) lv = l;
  volatile decltype(ml) mlv = ml;
  lv(); // expected-error{{no matching function for call to object of type}}
  mlv(); // expected-error{{no matching function for call to object of type}}

  bogus_override_if_virtual<decltype(l)> bogus;
}

// Default arguments (8.3.6) shall not be specified in the
// parameter-declaration-clause of a lambda- declarator.
// Note: Removed by core issue 974.
int test_default_args() {
  return [](int i = 5,  // expected-warning{{C++11 forbids default arguments for lambda expressions}}
            int j = 17) { return i+j;}(5, 6);
}

// Any exception-specification specified on a lambda-expression
// applies to the corresponding function call operator.
void test_exception_spec() {
  auto tl1 = []() throw(int) {};
  auto tl2 = []() {};
  static_assert(!noexcept(tl1()), "lambda can throw");
  static_assert(!noexcept(tl2()), "lambda can throw");

  auto ntl1 = []() throw() {};
  auto ntl2 = []() noexcept(true) {};
  auto ntl3 = []() noexcept {};
  static_assert(noexcept(ntl1()), "lambda cannot throw");  
  static_assert(noexcept(ntl2()), "lambda cannot throw");  
  static_assert(noexcept(ntl3()), "lambda cannot throw");  
}

