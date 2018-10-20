// RUN: %clang_cc1 -std=c++1y %s -verify -emit-llvm-only

namespace variadic_expansion {
  int f(int &, char &) { return 0; }
  template<class ... Ts> char fv(Ts ... ts) { return 0; }
  // FIXME: why do we get 2 error messages
  template <typename ... T> void g(T &... t) { //expected-note3{{declared here}}
    f([&a(t)]()->decltype(auto) {
      return a;
    }() ...);
    
    auto L = [x = f([&a(t)]()->decltype(auto) { return a; }()...)]() { return x; };
    const int y = 10;
    auto M = [x = y, 
                &z = y](T& ... t) { }; 
    auto N = [x = y, 
                &z = y, n = f(t...), 
                o = f([&a(t)](T& ... t)->decltype(auto) { return a; }(t...)...), t...](T& ... s) { 
                  fv([&a(t)]()->decltype(auto) { 
                    return a;
                  }() ...);
                };                 
    auto N2 = [x = y,                     //expected-note3{{begins here}}
                &z = y, n = f(t...), 
                o = f([&a(t)](T& ... t)->decltype(auto) { return a; }(t...)...)](T& ... s) { 
                  fv([&a(t)]()->decltype(auto) { //expected-error 3{{captured}}
                    return a;
                  }() ...);
                };                 

  }

  void h(int i, char c) { g(i, c); } //expected-note{{in instantiation}}
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
      return a;;
    };
        
  }
  {
    const int x = 10;
    auto L = [k = x](char a) { //expected-note {{declared}}
      return [](int b) { //expected-note {{begins}}
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
      return a;;
    };
        
  }
  { // will need to capture x in outer lambda
    const T x = 10; //expected-note {{declared}}
    auto L = [z = x](char a) { //expected-note {{begins}}
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
    auto L = [z = x](char a) { //expected-note {{begins}}
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

int run = test(); //expected-note {{instantiation}}

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
  f<int>();
  h<int>();
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
  auto a = [x = y]{}; // expected-error{{use of undeclared identifier 'y'}}

  int typo_foo; // expected-note 2 {{'typo_foo' declared here}}
  auto b = [x = typo_boo]{}; // expected-error{{use of undeclared identifier 'typo_boo'; did you mean 'typo_foo'}}
  auto c = [x(typo_boo)]{}; // expected-error{{use of undeclared identifier 'typo_boo'; did you mean 'typo_foo'}}
}
