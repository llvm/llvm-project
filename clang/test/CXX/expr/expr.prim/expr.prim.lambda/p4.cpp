// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify -DCPP1Y

void missing_lambda_declarator() {
  [](){}();
}

template<typename T> T get();

void infer_void_return_type(int i) {
  if (i > 17)
    return []() { }();

  if (i > 11)
    return []() { return; }();

  return [](int x) {
    switch (x) {
    case 0: return get<void>();
    case 1: return;
    case 2: return { 1, 2.0 }; //expected-error{{cannot deduce}}
    }
  }(7);
}

struct X { };

X infer_X_return_type(X x) {
  return [&x](int y) {
    if (y > 0)
      return X();
    else
      return x;
  }(5);
}

X infer_X_return_type_2(X x) {
  return [x](int y) {
    if (y > 0)
      return X();
    else
      return x; // ok even in c++11, per dr1048.
  }(5);
}

struct Incomplete; // expected-note 2{{forward declaration of 'Incomplete'}}
void test_result_type(int N) {  // expected-note {{declared here}}
  auto l1 = [] () -> Incomplete { }; // expected-error{{incomplete result type 'Incomplete' in lambda expression}}

  typedef int vla[N]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                         expected-note {{function parameter 'N' with unknown value cannot be used in a constant expression}}
  auto l2 = [] () -> vla { }; // expected-error{{function cannot return array type 'vla' (aka 'int[N]')}}
}

template <typename T>
void test_result_type_tpl(int N) { // expected-note 2{{declared here}}
  auto l1 = []() -> T {}; // expected-error{{incomplete result type 'Incomplete' in lambda expression}}
                          // expected-note@-1{{while substituting into a lambda expression here}}
  typedef int vla[N]; // expected-warning 2{{variable length arrays in C++ are a Clang extension}} \
                         expected-note 2{{function parameter 'N' with unknown value cannot be used in a constant expression}}
  auto l2 = []() -> vla {}; // expected-error{{function cannot return array type 'vla' (aka 'int[N]')}}
}

void test_result_type_call() {
  test_result_type_tpl<Incomplete>(10); // expected-note 2{{requested here}}
}
