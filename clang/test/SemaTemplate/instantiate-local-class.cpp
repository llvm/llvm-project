// RUN: %clang_cc1 -verify -std=c++11 %s
// expected-no-diagnostics
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
