// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
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

  bar<> foo{0};

  struct baz : public bar<> {
    using bar::bar;
  };

  baz data{0};
}

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

namespace PR13986 {
  constexpr unsigned Dynamic = 0;
  template <unsigned> class A { template <unsigned = Dynamic> void m_fn1(); };
  class Test {
    ~Test() {}
    A<1> m_target;
  };
}

// Template B is instantiated during checking if defaulted A copy constructor
// is constexpr. For this we check if S<int> copy constructor is constexpr. And
// for this we check S constructor template with default argument that mentions
// template B. In  turn, template instantiation triggers checking defaulted
// members exception spec. The problem is that it checks defaulted members not
// for instantiated class only, but all defaulted members so far. In this case
// we try to check exception spec for A default constructor which requires
// initializer for the field _a. But initializers are added after constexpr
// check so we reject the code because cannot find _a initializer.
namespace rdar34167492 {
  template <typename T> struct B { using type = bool; };

  template <typename T> struct S {
    S() noexcept;

    template <typename U, typename B<U>::type = true>
    S(const S<U>&) noexcept;
  };

  class A {
    A() noexcept = default;
    A(const A&) noexcept = default;
    S<int> _a{};
  };
}

namespace use_of_earlier_param {
  template<typename T> void f(T a, int = decltype(a)());
  void g() { f(0); }
}

#if __cplusplus >= 201402L
namespace lambda {
  // Verify that a default argument in a lambda can refer to the type of a
  // previous `auto` argument without crashing.
  template <class T>
  void bar() {
    (void) [](auto c, int x = sizeof(decltype(c))) {};
  }
  void foo() {
    bar<int>();
  }

#if __cplusplus >= 202002L
  // PR46648: ensure we don't reject this by triggering default argument
  // instantiation spuriously.
  auto x = []<typename T>(T x = 123) {};
  void y() { x(nullptr); }

  template<int A> struct X {
    template<int B> constexpr int f() {
      auto l = []<int C>(int n = A + B + C) { return n; };
      return l.template operator()<3>();
    }
  };
  static_assert(X<100>().f<20>() == 123);

  template<> template<int B> constexpr int X<200>::f() {
    auto l = []<int C>(int n = 300 + B + C) { return n; };
    return l.template operator()<1>();
  }
  static_assert(X<200>().f<20>() == 321);

  template<> template<> constexpr int X<300>::f<20>() {
    auto l = []<int C>(int n = 450 + C) { return n; };
    return l.template operator()<6>();
  }
  static_assert(X<300>().f<20>() == 456);
#endif
} // namespace lambda
#endif
