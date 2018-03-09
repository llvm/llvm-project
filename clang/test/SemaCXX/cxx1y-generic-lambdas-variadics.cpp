// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks %s
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s -DDELAYED_TEMPLATE_PARSING
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fms-extensions %s -DMS_EXTENSIONS
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing -fms-extensions %s -DMS_EXTENSIONS -DDELAYED_TEMPLATE_PARSING

namespace explicit_argument_variadics {


template<class ... Ts> void print(Ts ... ) { }

struct X { };
struct Y { };
struct Z { };

int test() { 
  {
    auto L = [](auto ... as) { };
    L.operator()<bool>(true);
  }
  {
    auto L = [](auto a) { };
    L.operator()<bool>(false);
  }
  {
    auto L = [](auto a, auto b) { };
    L.operator()<bool>(false, 'a');
  }
  {
    auto L = [](auto a, auto b) { };
    L.operator()<bool, char>(false, 'a');
  }
  {
    auto L = [](auto a, auto b, auto ... cs) { };
    L.operator()<bool, char>(false, 'a');
    L.operator()<bool, char, const char*>(false, 'a', "jim");    
  }

  {
    auto L = [](auto ... As) {
    };
    L.operator()<bool, double>(false, 3.14, "abc");
  }
  {
    auto L = [](auto A, auto B, auto ... As) {
    };
    L.operator()<bool>(false, 3.14, "abc");
    L.operator()<bool, char>(false, 3.14, "abc"); //expected-warning{{implicit conversion}}
    L.operator()<X, Y, bool, Z>(X{}, Y{}, 3.14, Z{}, X{}); //expected-warning{{implicit conversion}}
  }
  {
    auto L = [](auto ... As) {
      print("\nL::As = ", As ...);
      return [](decltype(As) ... as, auto ... Bs) {
        print("\nL::Inner::as = ", as ...);
        print("\nL::Inner::Bs = ", Bs ...);
        return 4;
      };
    };
    auto M = L.operator()<bool, double>(false, 3.14, "abc");
    M(false, 6.26, "jim", true);
    M.operator()<bool>(true, 6.26, "jim", false, 3.14);
  }
  {
    auto L = [](auto A, auto ... As) {
      print("\nL::As = ", As ...);
      return [](decltype(As) ... as, decltype(A) a, auto ... Bs) {
        print("\nL::Inner::as = ", as ...);
        print("\nL::Inner::Bs = ", Bs ...);
        return 4;
      };
    };
    auto M = L.operator()<bool, double>(false, 3.14, "abc");
    M(6.26, "jim", true);
    M.operator()<X>(6.26, "jim", false, X{}, Y{}, Z{});
  }
  
  return 0;
}
 int run = test();
} // end ns explicit_argument_extension



#ifdef PR18499_FIXED
namespace variadic_expansion {
  void f(int &, char &);

  template <typename ... T> void g(T &... t) {
    f([&a(t)]()->decltype(auto) {
      return a;
    }() ...);
    f([&a(f([&b(t)]()->decltype(auto) { return b; }()...), t)]()->decltype(auto) {
      return a;
    }()...);
  }

  void h(int i, char c) { g(i, c); }
}
#endif

namespace PR33082 {
  template<int ...I> void a() {
    int arr[] = { [](auto ...K) { (void)I; } ... }; // expected-error {{no viable conversion}} expected-note {{candidate}}
  }

  template<typename ...T> struct Pack {};
  template<typename ...T, typename ...U> void b(Pack<U...>, T ...t) {
    int arr[] = {[t...]() { // expected-error 2{{cannot initialize an array element of type 'int' with}}
      U u;
      return u;
    }()...};
  }

  void c() {
    int arr[] = {[](auto... v) {
      v; // expected-error {{unexpanded parameter pack 'v'}}
    }...}; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
  }

  void run() {
    a<1>(); // expected-note {{instantiation of}}
    b(Pack<int*, float*>(), 1, 2, 3); // expected-note {{instantiation of}}
  }
}
