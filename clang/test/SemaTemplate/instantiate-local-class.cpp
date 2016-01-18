// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -verify -std=c++11 -fdelayed-template-parsing %s

template<typename T>
void f0() {
  struct X;
  typedef struct Y {
    T (X::* f1())(int) { return 0; }
  } Y2;

  Y2 y = Y();
}

template void f0<int>();

// PR5764
namespace PR5764 {
  struct X {
    template <typename T>
    void Bar() {
      typedef T ValueType;
      struct Y {
        Y() { V = ValueType(); }

        ValueType V;
      };

      Y y;
    }
  };

  void test(X x) {
    x.Bar<int>();
  }
}

// Instantiation of local classes with virtual functions.
namespace local_class_with_virtual_functions {
  template <typename T> struct X { };
  template <typename T> struct Y { };

  template <typename T>
  void f() {
    struct Z : public X<Y<T>*> {
      virtual void g(Y<T>* y) { }
      void g2(int x) {(void)x;}
    };
    Z z;
    (void)z;
  }

  struct S { };
  void test() { f<S>(); }
}

namespace PR8801 {
  template<typename T>
  void foo() {
    class X;
    typedef int (X::*pmf_type)();
    class X : public T { };
    
    pmf_type pmf = &T::foo;
  }

  struct Y { int foo(); };

  template void foo<Y>();
}

namespace TemplatePacksAndLambdas {
  template <typename ...T> int g(T...);
  struct S {
    template <typename ...T> static void f(int f = g([]{ static T t; return ++t; }()...)) {}
  };
  void h() { S::f<int, int, int>(); }
}

namespace PR9685 {
  template <class Thing> void forEach(Thing t) { t.func(); }

  template <typename T> void doIt() {
    struct Functor {
      void func() { (void)i; }
      int i;
    };

    forEach(Functor());
  }

  void call() {
    doIt<int>();
  }
}

namespace PR12702 {
  struct S {
    template <typename F> bool apply(F f) { return f(); }
  };

  template <typename> struct T {
    void foo() {
      struct F {
        int x;

        bool operator()() { return x == 0; }
      };

      S().apply(F());
    }
  };

  void call() { T<int>().foo(); }
}

namespace PR17139 {
  template <class T> void foo(const T &t) { t.foo(); }

  template <class F> void bar(F *f) {
    struct B {
      F *fn;
      void foo() const { fn(); }
    } b = { f };
    foo(b);
  }

  void go() {}

  void test() { bar(go); }
}

namespace PR17740 {
class C {
public:
  template <typename T> static void foo(T function);
  template <typename T> static void bar(T function);
  template <typename T> static void func(T function);
};

template <typename T> void C::foo(T function) { function(); }

template <typename T> void C::bar(T function) {
  foo([&function]() { function(); });
}

template <typename T> void C::func(T function) {
  struct Struct {
    T mFunction;

    Struct(T function) : mFunction(function) {};

    void operator()() {
      mFunction();
    };
  };

  bar(Struct(function));
}

void call() {
  C::func([]() {});
}
}

namespace PR14373 {
  struct function {
    template <typename _Functor> function(_Functor __f) { __f(); }
  };
  template <typename Func> function exec_func(Func f) {
    struct functor {
      functor(Func f) : func(f) {}
      void operator()() const { func(); }
      Func func;
    };
    return functor(f);
  }
  struct Type {
    void operator()() const {}
  };
  int call() {
    exec_func(Type());
    return 0;
  }
}

namespace PR18907 {
template <typename>
class C : public C<int> {}; // expected-error{{within its own definition}}

template <typename X>
void F() {
  struct A : C<X> {};
}

struct B {
  void f() { F<int>(); }
};
}

namespace PR23194 {
  struct X {
    int operator()() const { return 0; }
  };
  struct Y {
    Y(int) {}
  };
  template <bool = true> int make_seed_pair() noexcept {
    struct state_t {
      X x;
      Y y{x()};
    };
    return 0;
  }
  int func() {
    return make_seed_pair();
  }
}

namespace PR18653 {
  // Forward declarations

  template<typename T> void f1() {
    void g1(struct x1);
    struct x1 {};
  }
  template void f1<int>();

  template<typename T> void f1a() {
    void g1(union x1);
    union x1 {};
  }
  template void f1a<int>();

  template<typename T> void f2() {
    void g2(enum x2);  // expected-error{{ISO C++ forbids forward references to 'enum' types}}
    enum x2 { nothing };
  }
  template void f2<int>();

  template<typename T> void f3() {
    void g3(enum class x3);
    enum class x3 { nothing };
  }
  template void f3<int>();


  template<typename T> void f4() {
    void g4(struct x4 {} x);  // expected-error{{'x4' cannot be defined in a parameter type}}
  }
  template void f4<int>();

  template<typename T> void f4a() {
    void g4(union x4 {} x);  // expected-error{{'x4' cannot be defined in a parameter type}}
  }
  template void f4a<int>();


  template <class T> void f();
  template <class T> struct S1 {
    void m() {
      f<class newclass>();
      f<union newunion>();
    }
  };
  template struct S1<int>;

  template <class T> struct S2 {
    void m() {
      f<enum new_enum>();  // expected-error{{ISO C++ forbids forward references to 'enum' types}}
    }
  };
  template struct S2<int>;

  template <class T> struct S3 {
    void m() {
      f<enum class new_enum>();
    }
  };
  template struct S3<int>;

  template <class T> struct S4 {
    struct local {};
    void m() {
      f<local>();
    }
  };
  template struct S4<int>;

  template <class T> struct S4a {
    union local {};
    void m() {
      f<local>();
    }
  };
  template struct S4a<int>;

  template <class T> struct S5 {
    enum local { nothing };
    void m() {
      f<local>();
    }
  };
  template struct S5<int>;

  template <class T> struct S7 {
    enum class local { nothing };
    void m() {
      f<local>();
    }
  };
  template struct S7<int>;


  template <class T> void fff(T *x);
  template <class T> struct S01 {
    struct local { };
    void m() {
      local x;
      fff(&x);
    }
  };
  template struct S01<int>;

  template <class T> struct S01a {
    union local { };
    void m() {
      local x;
      fff(&x);
    }
  };
  template struct S01a<int>;

  template <class T> struct S02 {
    enum local { nothing };
    void m() {
      local x;
      fff(&x);
    }
  };
  template struct S02<int>;

  template <class T> struct S03 {
    enum class local { nothing };
    void m() {
      local x;
      fff(&x);
    }
  };
  template struct S03<int>;


  template <class T> struct S04 {
    void m() {
      struct { } x;
      fff(&x);
    }
  };
  template struct S04<int>;

  template <class T> struct S04a {
    void m() {
      union { } x;
      fff(&x);
    }
  };
  template struct S04a<int>;

  template <class T> struct S05 {
    void m() {
      enum { nothing } x;
      fff(&x);
    }
  };
  template struct S05<int>;

  template <class T> struct S06 {
    void m() {
      class { virtual void mmm() {} } x;
      fff(&x);
    }
  };
  template struct S06<int>;
}

namespace PR20625 {
template <typename T>
void f() {
  struct N {
    static constexpr int get() { return 42; }
  };
  constexpr int n = N::get();
  static_assert(n == 42, "n == 42");
}

void g() { f<void>(); }
}


namespace PR21332 {
  template<typename T> void f1() {
    struct S {  // expected-note{{in instantiation of member class 'S' requested here}}
      void g1(int n = T::error);  // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
    };
  }
  template void f1<int>();  // expected-note{{in instantiation of function template specialization 'PR21332::f1<int>' requested here}}

  template<typename T> void f2() {
    struct S {  // expected-note{{in instantiation of member class 'S' requested here}}
      void g2() noexcept(T::error);  // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
    };
  }
  template void f2<int>();  // expected-note{{in instantiation of function template specialization 'PR21332::f2<int>' requested here}}

  template<typename T> void f3() {
    enum S {
      val = T::error;  // expected-error{{expected '}' or ','}} expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
    };
  }
  template void f3<int>();  //expected-note{{in instantiation of function template specialization 'PR21332::f3<int>' requested here}}

  template<typename T> void f4() {
    enum class S {
      val = T::error;  // expected-error{{expected '}' or ','}} expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
    };
  }
  template void f4<int>();  // expected-note{{in instantiation of function template specialization 'PR21332::f4<int>' requested here}}

  template<typename T> void f5() {
    class S {  // expected-note {{in instantiation of default member initializer 'PR21332::f5()::S::val' requested here}}
      int val = T::error;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
     };
  }
  template void f5<int>();  // expected-note {{in instantiation of function template specialization 'PR21332::f5<int>' requested here}}

  template<typename T> void f6() {
    class S {  // expected-note {{in instantiation of member function 'PR21332::f6()::S::get' requested here}}
      void get() {
        class S2 {  // expected-note {{in instantiation of member class 'S2' requested here}}
          void g1(int n = T::error);  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
        };
      }
    };
  }
  template void f6<int>();  // expected-note{{in instantiation of function template specialization 'PR21332::f6<int>' requested here}}

  template<typename T> void f7() {
    struct S { void g() noexcept(undefined_val); };  // expected-error{{use of undeclared identifier 'undefined_val'}}
  }
  template void f7<int>();
}
