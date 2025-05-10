// RUN: %clang_cc1 -std=c++1y %s -verify -emit-llvm-only
// RUN: %clang_cc1 -std=c++1z %s -verify -emit-llvm-only

namespace variadic_expansion {
  int f(int &, char &) { return 0; }
  template<class ... Ts> char fv(Ts ... ts) { return 0; }
  template <typename ... T> void g(T &... t) {
    f([&a(t)]()->decltype(auto) { // Line 8
      return a;
    }() ...); // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
    
    auto L = [x = undeclared_var()]() { return x; }; // expected-error {{use of undeclared identifier 'undeclared_var'}}
    const int y = 10;
    auto M = [x = y, 
                &z = y](T& ... t) { }; 
    auto N = [x = y, 
                &z = y, n = f(t...), 
                o = undeclared_var(), t...](T& ... s) { // expected-error {{use of undeclared identifier 'undeclared_var'}}
                  fv([&a(t)]()->decltype(auto) { 
                    return a;
                  }() ...); // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
                };                 
    auto N2 = [x = y, 
               &z = y, n = f(t...), 
               o = undeclared_var(), &t...](T& ... s) { // expected-error {{use of undeclared identifier 'undeclared_var'}}
                  fv([&a(t), &t...]() -> decltype(auto) { 
                    return a;
                  }() ...); // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
                };
  }

  void h(int i, char c) { g(i, c); }
}

namespace odr_use_within_init_capture {

int test() {
  
  { // no captures
    const int x = 10;
    auto L = [z = x + 2](int a) {
      auto M = [y = x - 2](char b) {
        return y;
      };
      return M;
    };
        
  }
  { // should not capture
    const int x = 10;
    auto L = [&z = x](int a) {
      return a;
    };
        
  }
  {
    const int x = 10;
    auto L = [k = x](char a) {  //expected-note {{declared}}
      return [](int b) {        //expected-note {{begins}} expected-note 2 {{capture 'k' by}} expected-note 2 {{default capture by}}
        return [j = k](int c) { //expected-error {{cannot be implicitly captured}}
          return c;
        };
      };
    };
  }
  {
    const int x = 10;
    auto L = [k = x](char a) { 
      return [=](int b) { 
        return [j = k](int c) { 
          return c;
        };
      };
    };
  }
  {
    const int x = 10;
    auto L = [k = x](char a) { 
      return [k](int b) { 
        return [j = k](int c) { 
          return c;
        };
      };
    };
  }

  return 0;
}

int run = test();

}

namespace odr_use_within_init_capture_template {

template<class T = int>
int test(T t = T{}) {

  { // no captures
    const T x = 10;
    auto L = [z = x](char a) {
      auto M = [y = x](T b) {
        return y;
      };
      return M;
    };
        
  }
  { // should not capture
    const T x = 10;
    auto L = [&z = x](T a) {
      return a;
    };
        
  }
  { // will need to capture x in outer lambda
    const T x = 10; //expected-note {{declared}}
    auto L = [z = x](char a) { //expected-note {{begins}} expected-note 2 {{capture 'x' by}} expected-note 2 {{default capture by}} expected-note {{substituting into a lambda}}
      auto M = [&y = x](T b) { //expected-error {{cannot be implicitly captured}}
        return y;
      };
      return M;
    };
  }
  { // will need to capture x in outer lambda
    const T x = 10; 
    auto L = [=,z = x](char a) { 
      auto M = [&y = x](T b) { 
        return y;
      };
      return M;
    };
        
  }
  { // will need to capture x in outer lambda
    const T x = 10; 
    auto L = [x, z = x](char a) { 
      auto M = [&y = x](T b) { 
        return y;
      };
      return M;
    };
  }
  { // will need to capture x in outer lambda
    const int x = 10; //expected-note {{declared}}
    auto L = [z = x](char a) { //expected-note {{begins}} expected-note 2 {{capture 'x' by}} expected-note 2 {{default capture by}} expected-note {{substituting into a lambda}}
      auto M = [&y = x](T b) { //expected-error {{cannot be implicitly captured}}
        return y;
      };
      return M;
    };
  }
  {
    // no captures
    const T x = 10;
    auto L = [z = 
                  [z = x, &y = x](char a) { return z + y; }('a')](char a) 
      { return z; };
  
  }
  
  return 0;
}

int run = test(); //expected-note 2 {{instantiation}}

}

namespace classification_of_captures_of_init_captures {

template <typename T>
void f() {
  [a = 24] () mutable {
    [&a] { a = 3; }();
  }();
}

template <typename T>
void h() {
  [a = 24] (auto param) mutable {
    [&a] { a = 3; }();
  }(42);
}

int run() {
  return 0;
}

}

namespace N3922 {
  struct X { X(); explicit X(const X&); int n; };
  auto a = [x{X()}] { return x.n; }; // ok
  auto b = [x = {X()}] {}; // expected-error{{<initializer_list>}}
}

namespace init_capture_non_mutable {
void test(double weight) {
  double init;
  auto find = [max = init](auto current) {
    max = current; // expected-error{{cannot assign to a variable captured by copy in a non-mutable lambda}}
  };
  find(weight); // expected-note {{in instantiation of function template specialization}}
}
}

namespace init_capture_undeclared_identifier {
  auto a = [x = y]{}; // expected-error{{use of undeclared identifier 'y'}} expected-error{{invalid initializer type for lambda capture}}

  int typo_foo; // expected-note 2 {{'typo_foo' declared here}}
  auto b = [x = typo_boo]{}; // expected-error{{use of undeclared identifier 'typo_boo'; did you mean 'typo_foo'}}
  auto c = [x(typo_boo)]{}; // expected-error{{use of undeclared identifier 'typo_boo'; did you mean 'typo_foo'}}
}

namespace copy_evasion {
  struct A {
    A();
    A(const A&) = delete;
  };
  auto x = [a{A()}] {};
#if __cplusplus >= 201702L
  // ok, does not copy an 'A'
#else
  // expected-error@-4 {{call to deleted}}
  // expected-note@-7 {{deleted}}
#endif
}
