template<typename T> struct S {
  __attribute__((always_inline)) static int f() { return 0; }
  __attribute__((always_inline, visibility("hidden"))) static int g() { return 0; }
};

extern template struct S<int>;

template<typename T> T min(T a, T b) { return a < b ? a : b; }

extern decltype(min(1, 2)) instantiate_min_decl;

template<typename T> struct CtorInit {
  static int f() { return 0; }
  int a;
  CtorInit() : a(f()) {}
};

namespace ImplicitSpecialMembers {
  struct A {
    A(const A&);
  };
  struct B {
    A a;
    B(int);
  };
  struct C {
    A a;
    C(int);
  };
  struct D {
    A a;
    D(int);
  };
}

namespace OperatorDeleteLookup {
  struct A { void operator delete(void*); virtual ~A() = default; };
  template<typename T> struct B { void operator delete(void*); virtual ~B() {} typedef int t; };
  typedef B<int>::t b_int_instantiated;
}

namespace EmitInlineMethods {
  struct A {
    void f() {}
    void g();
  };
  struct B {
    void f();
    void g() {}
  };
}
