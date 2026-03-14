// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y -DCXX1Y 

namespace test_factorial {

auto Fact = [](auto Self, unsigned n) -> unsigned {
    return !n ? 1 : Self(Self, n - 1) * n;
};

auto six = Fact(Fact, 3);

}

namespace overload_generic_lambda {
  template <class F1, class F2> struct overload : F1, F2 {
    using F1::operator();
    using F2::operator();
    overload(F1 f1, F2 f2) : F1(f1), F2(f2) { }
  };

  auto NumParams = [](auto Self, auto h, auto ... rest) -> unsigned {
    return 1 + Self(Self, rest...);
  };
  auto Base = [](auto Self, auto h) -> unsigned {
      return 1;
  };
  overload<decltype(Base), decltype(NumParams)> O(Base, NumParams);
  int num_params =  O(O, 5, 3, "abc", 3.14, 'a');
}


namespace overload_generic_lambda_return_type_deduction {
  template <class F1, class F2> struct overload : F1, F2 {
    using F1::operator();
    using F2::operator();
    overload(F1 f1, F2 f2) : F1(f1), F2(f2) { }
  };

  auto NumParams = [](auto Self, auto h, auto ... rest) {
    return 1 + Self(Self, rest...);
  };
  auto Base = [](auto Self, auto h) {
      return 1;
  };
  overload<decltype(Base), decltype(NumParams)> O(Base, NumParams);
  int num_params =  O(O, 5, 3, "abc", 3.14, 'a');
}

namespace test_standard_p5 {
// FIXME: This test should eventually compile without an explicit trailing return type
auto glambda = [](auto a, auto&& b) ->bool { return a < b; };
bool b = glambda(3, 3.14); // OK

}
namespace test_deduction_failure {
 int test() {
   auto g = [](auto *a) { //expected-note{{candidate template ignored}}
    return a;
   };
   struct X { };
   X *x;
   g(x);
   g(3); //expected-error{{no matching function}}
   return 0;
 }

}
  
namespace test_instantiation_or_sfinae_failure {
int test2() {
  {
    auto L = [](auto *a) { 
                return (*a)(a); }; //expected-error{{called object type 'double' is not a function}}
    double d;
    L(&d); //expected-note{{in instantiation of}}
    auto M = [](auto b) { return b; };
    L(&M); // ok
  }
  {
    auto L = [](auto *a) ->decltype (a->foo()) { //expected-note2{{candidate template ignored:}}
                return (*a)(a); }; 
    double d;
    L(&d); //expected-error{{no matching function for call}} 
    auto M = [](auto b) { return b; };
    L(&M); //expected-error{{no matching function for call}} 
 
  }
  return 0;
}


}
  
namespace test_misc {
auto GL = [](auto a, decltype(a) b) //expected-note{{candidate function}} 
                -> int { return a + b; };

void test() {
   struct X { };
   GL(3, X{}); //expected-error{{no matching function}}
}

void test2() {
  auto l = [](auto *a) -> int { 
              (*a)(a); return 0; }; //expected-error{{called object type 'double' is not a function}}
  l(&l);
  double d;
  l(&d); //expected-note{{in instantiation of}}
}

}

namespace nested_lambdas {
  int test() {
    auto L = [](auto a) {
                 return [=](auto b) {  
                           return a + b;
                        };
              };
  }
  auto get_lambda() {
    return [](auto a) {
      return a; 
    };
  };
  
  int test2() {
    auto L = get_lambda();
    L(3);
  }
}

