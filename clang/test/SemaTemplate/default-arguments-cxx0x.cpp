// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

// Test default template arguments for function templates.
template<typename T = int>
void f0();

template<typename T>
void f0();

void g0() {
  f0(); // okay!
} 

template<typename T, int N = T::value>
int &f1(T);

float &f1(...);

struct HasValue {
  static const int value = 17;
};

void g1() {
  float &fr = f1(15);
  int &ir = f1(HasValue());
}

namespace PR16689 {
  template <typename T1, typename T2> class tuple {
  public:
      template <typename = T2>
      constexpr tuple() {}
  };
  template <class X, class... Y> struct a : public X {
    using X::X;
  };
  auto x = a<tuple<int, int> >();
}

namespace PR16975 {
  template <typename...> struct is {
    constexpr operator bool() const { return false; }
  };

  template <typename... Types>
  struct bar {
    template <typename T,
              bool = is<Types...>()>
    bar(T);
  };

  struct baz : public bar<> {
    using bar::bar;
  };

  baz data{0};
}

// rdar://23810407
// An IRGen failure due to a symbol collision due to a default argument
// being instantiated twice.  Credit goes to Richard Smith for this
// reduction to a -fsyntax-only failure.
namespace rdar23810407 {
  // Instantiating the default argument multiple times will produce two
  // different lambda types and thus instantiate this function multiple
  // times, which will produce conflicting extern variable declarations.
  template<typename T> int f(T t) {
    extern T rdar23810407_variable;
    return 0;
  }
  template<typename T> int g(int a = f([] {}));
  void test() {
    g<int>();
    g<int>();
  }
}

// rdar://problem/24480205
namespace PR13986 {
  constexpr unsigned Dynamic = 0;
  template <unsigned> class A { template <unsigned = Dynamic> void m_fn1(); };
  class Test {
    ~Test() {}
    A<1> m_target;
  };
}
