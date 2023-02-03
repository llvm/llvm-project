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

struct Incomplete; // expected-note{{forward declaration of 'Incomplete'}}
void test_result_type(int N) {
  auto l1 = [] () -> Incomplete { }; // expected-error{{incomplete result type 'Incomplete' in lambda expression}}

  typedef int vla[N];
  auto l2 = [] () -> vla { }; // expected-error{{function cannot return array type 'vla' (aka 'int[N]')}}
}
