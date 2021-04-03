// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify

int a;
int &b = [] (int &r) -> decltype(auto) { return r; } (a);
int &c = [] (int &r) -> decltype(auto) { return (r); } (a);
int &d = [] (int &r) -> auto & { return r; } (a);
int &e = [] (int &r) -> auto { return r; } (a); // expected-error {{cannot bind to a temporary}}
int &f = [] (int r) -> decltype(auto) { return r; } (a); // expected-error {{cannot bind to a temporary}}
int &g = [] (int r) -> decltype(auto) { return (r); } (a); // expected-warning {{reference to stack}}


int test_explicit_auto_return()
{
    struct X {};
    auto L = [](auto F, auto a) { return F(a); };
    auto M = [](auto a) -> auto { return a; }; // OK
    auto MRef = [](auto b) -> auto& { return b; }; //expected-warning{{reference to stack}}
    auto MPtr = [](auto c) -> auto* { return &c; }; //expected-warning{{address of stack}}
    auto MDeclType = [](auto&& d) -> decltype(auto) { return static_cast<decltype(d)>(d); }; //OK
    M(3);

    auto &&x = MDeclType(X{});
    auto &&x1 = M(X{});
    auto &&x2 = MRef(X{});//expected-note{{in instantiation of}}
    auto &&x3 = MPtr(X{}); //expected-note{{in instantiation of}}
    return 0;
}

int test_implicit_auto_return()
{
  {
    auto M = [](auto a) { return a; };
    struct X {};
    X x = M(X{});

  }
}

int test_multiple_returns()  {
    auto M = [](auto a) {
      bool k;
      if (k)
        return a;
      else
        return 5; //expected-error{{deduced as 'int' here}}
    };
    M(3); // OK
    M('a'); //expected-note{{in instantiation of}}
  return 0;
}
int test_no_parameter_list()
{
  static int si = 0;
    auto M = [] { return 5; }; // OK
    auto M2 = [] -> auto && { return si; };
#if __cplusplus <= 202002L
      // expected-warning@-2{{is a C++2b extension}}
#endif
    M();
}

int test_conditional_in_return() {
  auto Fac = [](auto f, auto n) {
    return n <= 0 ? n : f(f, n - 1) * n;
  };
  // FIXME: this test causes a recursive limit - need to error more gracefully.
  //Fac(Fac, 3);

}