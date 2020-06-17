// RUN: %clang_cc1 -std=c++14 -Wno-unused-value -fsyntax-only -verify -verify=expected-cxx14 -fblocks %s
// RUN: %clang_cc1 -std=c++17 -Wno-unused-value -fsyntax-only -verify -fblocks %s

namespace std { class type_info; };

namespace ExplicitCapture {
  class C {
    int Member;

    static void Overload(int);
    void Overload();
    virtual C& Overload(float);

    void ImplicitThisCapture() {
      [](){(void)Member;}; // expected-error {{'this' cannot be implicitly captured in this context}}
      const int var = [](){(void)Member; return 0;}(); // expected-error {{'this' cannot be implicitly captured in this context}}
      [&](){(void)Member;};

      [this](){(void)Member;};
      [this]{[this]{};};
      []{[this]{};};// expected-error {{'this' cannot be implicitly captured in this context}}
      []{Overload(3);};
      []{Overload();}; // expected-error {{'this' cannot be implicitly captured in this context}}
      []{(void)typeid(Overload());};
      []{(void)typeid(Overload(.5f));};// expected-error {{'this' cannot be implicitly captured in this context}}
    }
  };

  void f() {
    [this] () {}; // expected-error {{'this' cannot be captured in this context}}
  }
}

namespace ReturnDeduction {
  void test() {
    [](){ return 1; };
    [](){ return 1; };
    [](){ return ({return 1; 1;}); };
    [](){ return ({return 'c'; 1;}); }; // expected-error {{must match previous return type}}
    []()->int{ return 'c'; return 1; };
    [](){ return 'c'; return 1; };  // expected-error {{must match previous return type}}
    []() { return; return (void)0; };
    [](){ return 1; return 1; };
  }
}

namespace ImplicitCapture {
  void test() {
    int a = 0; // expected-note 5 {{declared}}
    []() { return a; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{begins here}}
    [&]() { return a; };
    [=]() { return a; };
    [=]() { int* b = &a; }; // expected-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'const int *'}}
    [=]() { return [&]() { return a; }; };
    []() { return [&]() { return a; }; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}
    []() { return ^{ return a; }; };// expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}
    []() { return [&a] { return a; }; }; // expected-error 2 {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note 2 {{lambda expression begins here}}
    [=]() { return [&a] { return a; }; }; //

    const int b = 2;
    []() { return b; };

    union { // expected-note {{declared}}
      int c;
      float d;
    };
    d = 3;
    [=]() { return c; }; // expected-error {{unnamed variable cannot be implicitly captured in a lambda expression}}

    __block int e; // expected-note 2{{declared}}
    [&]() { return e; }; // expected-error {{__block variable 'e' cannot be captured in a lambda expression}}
    [&e]() { return e; }; // expected-error {{__block variable 'e' cannot be captured in a lambda expression}}

    int f[10]; // expected-note {{declared}}
    [&]() { return f[2]; };
    (void) ^{ return []() { return f[2]; }; }; // expected-error {{variable 'f' cannot be implicitly captured in a lambda with no capture-default specified}} \
    // expected-note{{lambda expression begins here}}

    struct G { G(); G(G&); int a; }; // expected-note 6 {{not viable}}
    G g;
    [=]() { const G* gg = &g; return gg->a; };
    [=]() { return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error {{no matching constructor for initialization of 'G'}}
    (void)^{ return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error 2 {{no matching constructor for initialization of 'const G'}}

    const int h = a; // expected-note {{declared}}
    []() { return h; }; // expected-error {{variable 'h' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}

    // References can appear in constant expressions if they are initialized by
    // reference constant expressions.
    int i;
    int &ref_i = i; // expected-note {{declared}}
    [] { return ref_i; }; // expected-error {{variable 'ref_i' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}

    static int j;
    int &ref_j = j;
    [] { return ref_j; }; // ok
  }
}

namespace SpecialMembers {
  void f() {
    auto a = []{}; // expected-note 2{{here}} expected-note 2{{candidate}}
    decltype(a) b; // expected-error {{no matching constructor}}
    decltype(a) c = a;
    decltype(a) d = static_cast<decltype(a)&&>(a);
    a = a; // expected-error {{copy assignment operator is implicitly deleted}}
    a = static_cast<decltype(a)&&>(a); // expected-error {{copy assignment operator is implicitly deleted}}
  }
  struct P {
    P(const P&) = delete; //expected-note {{deleted here}} // expected-cxx14-note {{deleted here}}
  };
  struct Q {
    ~Q() = delete; // expected-note {{deleted here}}
  };
  struct R {
    R(const R&) = default;
    R(R&&) = delete;
    R &operator=(const R&) = delete;
    R &operator=(R&&) = delete;
  };
  void g(P &p, Q &q, R &r) {
    // FIXME: The note attached to the second error here is just amazingly bad.
    auto pp = [p]{}; // expected-error {{deleted constructor}} expected-cxx14-error {{deleted copy constructor of '(lambda}}
    // expected-cxx14-note@-1 {{copy constructor of '' is implicitly deleted because field '' has a deleted copy constructor}}
    auto qq = [q]{}; // expected-error {{deleted function}} expected-note {{because}}

    auto a = [r]{}; // expected-note 2{{here}}
    decltype(a) b = a;
    decltype(a) c = static_cast<decltype(a)&&>(a); // ok, copies R
    a = a; // expected-error {{copy assignment operator is implicitly deleted}}
    a = static_cast<decltype(a)&&>(a); // expected-error {{copy assignment operator is implicitly deleted}}
  }
}

namespace PR12031 {
  struct X {
    template<typename T>
    X(const T&);
    ~X();
  };

  void f(int i, X x);
  void g() {
    const int v = 10;
    f(v, [](){});
  }
}

namespace Array {
  int &f(int *p);
  char &f(...);
  void g() {
    int n = -1;
    [=] {
      int arr[n]; // VLA
    } ();

    const int m = -1;
    [] {
      int arr[m]; // expected-error{{negative size}}
    } ();

    [&] {
      int arr[m]; // expected-error{{negative size}}
    } ();

    [=] {
      int arr[m]; // expected-error{{negative size}}
    } ();

    [m] {
      int arr[m]; // expected-error{{negative size}}
    } ();
  }
}

void PR12248()
{
  unsigned int result = 0;
  auto l = [&]() { ++result; };
}

namespace ModifyingCapture {
  void test() {
    int n = 0;
    [=] {
      n = 1; // expected-error {{cannot assign to a variable captured by copy in a non-mutable lambda}}
    };
  }
}

namespace VariadicPackExpansion {
  template<typename T, typename U> using Fst = T;
  template<typename...Ts> bool g(Fst<bool, Ts> ...bools);
  template<typename...Ts> bool f(Ts &&...ts) {
    return g<Ts...>([&ts] {
      if (!ts)
        return false;
      --ts;
      return true;
    } () ...);
  }
  void h() {
    int a = 5, b = 2, c = 3;
    while (f(a, b, c)) {
    }
  }

  struct sink {
    template<typename...Ts> sink(Ts &&...) {}
  };

  template<typename...Ts> void local_class() {
    sink {
      [] (Ts t) {
        struct S : Ts {
          void f(Ts t) {
            Ts &that = *this;
            that = t;
          }
          Ts g() { return *this; };
        };
        S s;
        s.f(t);
        return s;
      } (Ts()).g() ...
    };
  };
  struct X {}; struct Y {};
  template void local_class<X, Y>();

  template<typename...Ts> void nested(Ts ...ts) {
    f(
      // Each expansion of this lambda implicitly captures all of 'ts', because
      // the inner lambda also expands 'ts'.
      [&] {
        return ts + [&] { return f(ts...); } ();
      } () ...
    );
  }
  template void nested(int, int, int);

  template<typename...Ts> void nested2(Ts ...ts) { // expected-note 2{{here}}
    // Capture all 'ts', use only one.
    f([&ts...] { return ts; } ()...);
    // Capture each 'ts', use it.
    f([&ts] { return ts; } ()...);
    // Capture all 'ts', use all of them.
    f([&ts...] { return (int)f(ts...); } ());
    // Capture each 'ts', use all of them. Ill-formed. In more detail:
    //
    // We instantiate two lambdas here; the first captures ts$0, the second
    // captures ts$1. Both of them reference both ts parameters, so both are
    // ill-formed because ts can't be implicitly captured.
    //
    // FIXME: This diagnostic does not explain what's happening. We should
    // specify which 'ts' we're referring to in its diagnostic name. We should
    // also say which slice of the pack expansion is being performed in the
    // instantiation backtrace.
    f([&ts] { return (int)f(ts...); } ()...); // \
    // expected-error 2{{'ts' cannot be implicitly captured}} \
    // expected-note 2{{lambda expression begins here}}
  }
  template void nested2(int); // ok
  template void nested2(int, int); // expected-note {{in instantiation of}}
}

namespace PR13860 {
  void foo() {
    auto x = PR13860UndeclaredIdentifier(); // expected-error {{use of undeclared identifier 'PR13860UndeclaredIdentifier'}}
    auto y = [x]() { };
    static_assert(sizeof(y), "");
  }
}

namespace PR13854 {
  auto l = [](void){};
}

namespace PR14518 {
  auto f = [](void) { return __func__; }; // no-warning
}

namespace PR16708 {
  auto L = []() {
    auto ret = 0;
    return ret;
    return 0;
  };
}

namespace TypeDeduction {
  struct S {};
  void f() {
    const S s {};
    S &&t = [&] { return s; } ();
#if __cplusplus > 201103L
    S &&u = [&] () -> auto { return s; } ();
#endif
  }
}


namespace lambdas_in_NSDMIs {
  template<class T>
  struct L {
      T t{};
      T t2 = ([](int a) { return [](int b) { return b; };})(t)(t);    
  };
  L<int> l; 
  
  namespace non_template {
    struct L {
      int t = 0;
      int t2 = ([](int a) { return [](int b) { return b; };})(t)(t);    
    };
    L l; 
  }
}

// PR18477: don't try to capture 'this' from an NSDMI encountered while parsing
// a lambda.
namespace NSDMIs_in_lambdas {
  template<typename T> struct S { int a = 0; int b = a; };
  void f() { []() { S<int> s; }; }

  auto x = []{ struct S { int n, m = n; }; };
  auto y = [&]{ struct S { int n, m = n; }; }; // expected-error {{non-local lambda expression cannot have a capture-default}}
  void g() { auto z = [&]{ struct S { int n, m = n; }; }; }
}

namespace CaptureIncomplete {
  struct Incomplete; // expected-note 2{{forward decl}}
  void g(const Incomplete &a);
  void f(Incomplete &a) {
    (void) [a] {}; // expected-error {{incomplete}}
    (void) [&a] {};

    (void) [=] { g(a); }; // expected-error {{incomplete}}
    (void) [&] { f(a); };
  }
}

namespace CaptureAbstract {
  struct S {
    virtual void f() = 0; // expected-note {{unimplemented}}
    int n = 0;
  };
  struct T : S {
    constexpr T() {}
    void f();
  };
  void f() {
    constexpr T t = T();
    S &s = const_cast<T&>(t);
    // FIXME: Once we properly compute odr-use per DR712, this should be
    // accepted (and should not capture 's').
    [=] { return s.n; }; // expected-error {{abstract}}
  }
}

namespace PR18128 {
  auto l = [=]{}; // expected-error {{non-local lambda expression cannot have a capture-default}}

  struct S {
    int n;
    int (*f())[true ? 1 : ([=]{ return n; }(), 0)];
    // expected-error@-1 {{non-local lambda expression cannot have a capture-default}}
    // expected-error@-2 {{invalid use of non-static data member 'n'}}
    // expected-cxx14-error@-3 {{a lambda expression may not appear inside of a constant expression}}
    int g(int k = ([=]{ return n; }(), 0));
    // expected-error@-1 {{non-local lambda expression cannot have a capture-default}}
    // expected-error@-2 {{invalid use of non-static data member 'n'}}

    int a = [=]{ return n; }(); // ok
    int b = [=]{ return [=]{ return n; }(); }(); // ok
    int c = []{ int k = 0; return [=]{ return k; }(); }(); // ok
    int d = []{ return [=]{ return n; }(); }(); // expected-error {{'this' cannot be implicitly captured in this context}}
  };
}

namespace PR18473 {
  template<typename T> void f() {
    T t(0);
    (void) [=]{ int n = t; }; // expected-error {{deleted}}
  }

  template void f<int>();
  struct NoCopy {
    NoCopy(int);
    NoCopy(const NoCopy &) = delete; // expected-note {{deleted}}
    operator int() const;
  };
  template void f<NoCopy>(); // expected-note {{instantiation}}
}

void PR19249() {
  auto x = [&x]{}; // expected-error {{cannot appear in its own init}}
}

namespace PR20731 {
template <class L, int X = sizeof(L)>
void Job(L l);

template <typename... Args>
void Logger(Args &&... args) {
  auto len = Invalid_Function((args)...);
  // expected-error@-1 {{use of undeclared identifier 'Invalid_Function'}}
  Job([len]() {});
}

void GetMethod() {
  Logger();
  // expected-note@-1 {{in instantiation of function template specialization 'PR20731::Logger<>' requested here}}
}

template <typename T>
struct A {
  T t;
  // expected-error@-1 {{field has incomplete type 'void'}}
};

template <typename F>
void g(F f) {
  auto a = A<decltype(f())>{};
  // expected-note@-1 {{in instantiation of template class 'PR20731::A<void>' requested here}}
  auto xf = [a, f]() {};
  int x = sizeof(xf);
};
void f() {
  g([] {});
  // expected-note-re@-1 {{in instantiation of function template specialization 'PR20731::g<(lambda at {{.*}}>' requested here}}
}

template <class _Rp> struct function {
  template <class _Fp>
  function(_Fp) {
    static_assert(sizeof(_Fp) > 0, "Type must be complete.");
  }
};

template <typename T> void p(T t) {
  auto l = some_undefined_function(t);
  // expected-error@-1 {{use of undeclared identifier 'some_undefined_function'}}
  function<void()>(([l]() {}));
}
void q() { p(0); }
// expected-note@-1 {{in instantiation of function template specialization 'PR20731::p<int>' requested here}}
}

namespace lambda_in_default_mem_init {
  template<typename T> void f() {
    struct S { int n = []{ return 0; }(); };
  }
  template void f<int>();

  template<typename T> void g() {
    struct S { int n = [](int n){ return n; }(0); };
  }
  template void g<int>();
}

namespace error_in_transform_prototype {
  template<class T>
  void f(T t) {
    // expected-error@+2 {{type 'int' cannot be used prior to '::' because it has no members}}
    // expected-error@+1 {{no member named 'ns' in 'error_in_transform_prototype::S'}}
    auto x = [](typename T::ns::type &k) {};
  }
  class S {};
  void foo() {
    f(5); // expected-note {{requested here}}
    f(S()); // expected-note {{requested here}}
  }
}

namespace PR21857 {
  template<typename Fn> struct fun : Fn {
    fun() = default;
    using Fn::operator();
  };
  template<typename Fn> fun<Fn> wrap(Fn fn);
  auto x = wrap([](){});
}

namespace PR13987 {
class Enclosing {
  void Method(char c = []()->char {
    int d = []()->int {
        struct LocalClass {
          int Method() { return 0; }
        };
      return 0;
    }();
    return d; }()
  );
};
}

namespace PR23860 {
template <class> struct A {
  void f(int x = []() {
    struct B {
      void g() {}
    };
    return 0;
  }());
};

int main() {
}

A<int> a;
}

// rdar://22032373
namespace rdar22032373 {
void foo() {
  auto blk = [](bool b) {
    if (b)
      return undeclared_error; // expected-error {{use of undeclared identifier}}
    return 0;
  };
}
}

namespace nested_lambda {
template <int N>
class S {};

void foo() {
  const int num = 18; 
  auto outer = []() {
    auto inner = [](S<num> &X) {};  
  };
}
}

namespace PR27994 {
struct A { template <class T> A(T); };

template <class T>
struct B {
  int x;
  A a = [&] { int y = x; };
  A b = [&] { [&] { [&] { int y = x; }; }; };
  A d = [&](auto param) { int y = x; };
  A e = [&](auto param) { [&] { [&](auto param2) { int y = x; }; }; };
};

B<int> b;

template <class T> struct C {
  struct D {
    int x;
    A f = [&] { int y = x; };
  };
};

int func() {
  C<int> a;
  decltype(a)::D b;
}
}

namespace PR30566 {
int name1; // expected-note {{'name1' declared here}}

struct S1 {
  template<class T>
  S1(T t) { s = sizeof(t); }
  int s;
};

void foo1() {
  auto s0 = S1{[name=]() {}}; // expected-error 2 {{expected expression}}
  auto s1 = S1{[name=name]() {}}; // expected-error {{use of undeclared identifier 'name'; did you mean 'name1'?}}
}
}

namespace PR25627_dont_odr_use_local_consts {
  
  template<int> struct X {};
  
  void foo() {
    const int N = 10;
    (void) [] { X<N> x; };
  }
}

namespace ConversionOperatorDoesNotHaveDeducedReturnType {
  auto x = [](int){};
  auto y = [](auto &v) -> void { v.n = 0; };
  using T = decltype(x);
  using U = decltype(y);
  using ExpectedTypeT = void (*)(int);
  template<typename T>
    using ExpectedTypeU = void (*)(T&);

  struct X {
#if __cplusplus > 201402L
    friend constexpr auto T::operator()(int) const;
    friend constexpr T::operator ExpectedTypeT() const noexcept;
#else
    friend auto T::operator()(int) const;
    friend T::operator ExpectedTypeT() const;
#endif

    // FIXME: The first of these should match. The second should not.
    template<typename T>
      friend void U::operator()(T&) const; // expected-error {{does not match}}
    template<typename T>
      friend U::operator ExpectedTypeU<T>() const; // expected-error {{does not match}}

  private:
    int n;
  };

  // Should be OK: lambda's call operator is a friend.
  void use(X &x) { y(x); }

  // This used to crash in return type deduction for the conversion opreator.
  struct A { int n; void f() { +[](decltype(n)) {}; } };
}

namespace TypoCorrection {
template <typename T> struct X {};
// expected-note@-1 {{template parameter is declared here}}

template <typename T>
void Run(const int& points) {
// expected-note@-1 {{'points' declared here}}
  auto outer_lambda = []() {
    auto inner_lambda = [](const X<Points>&) {};
    // expected-error@-1 {{use of undeclared identifier 'Points'; did you mean 'points'?}}
    // expected-error@-2 {{template argument for template type parameter must be a type}}
  };
}
}

void operator_parens() {
  [&](int x){ operator()(); }(0); // expected-error {{undeclared 'operator()'}}
}
