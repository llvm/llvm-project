// RUN: %clang_cc1 -std=c++0x -Wno-unused-value -fsyntax-only -verify -fblocks %s

namespace std { class type_info; };

namespace ExplicitCapture {
  class C {
    int Member;

    static void Overload(int);
    void Overload();
    virtual C& Overload(float);

    void ImplicitThisCapture() {
      [](){(void)Member;}; // expected-error {{'this' cannot be implicitly captured in this context}}
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

    __block int e; // expected-note 3 {{declared}}
    [&]() { return e; }; // expected-error {{__block variable 'e' cannot be captured in a lambda expression}}
    [&e]() { return e; }; // expected-error 2 {{__block variable 'e' cannot be captured in a lambda expression}}

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

namespace NullPtr {
  int &f(int *p);
  char &f(...);
  void g() {
    int n = 0;
    [=] {
      char &k = f(n); // not a null pointer constant
    } ();

    const int m = 0;
    [=] {
      int &k = f(m); // expected-warning{{expression which evaluates to zero treated as a null pointer constant of type 'int *'}}
    } ();

    [=] () -> bool {
      int &k = f(m); // expected-warning{{expression which evaluates to zero treated as a null pointer constant of type 'int *'}}
      return &m == 0;
    } ();

    [m] {
      int &k = f(m); // expected-warning{{expression which evaluates to zero treated as a null pointer constant of type 'int *'}}
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
