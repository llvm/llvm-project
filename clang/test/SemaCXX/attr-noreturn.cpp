// RUN: %clang_cc1 -fsyntax-only -verify %s

// Reachability tests have to come first because they get suppressed
// if any errors have occurred.
namespace test5 {
  struct A {
    __attribute__((noreturn)) void fail();
    void nofail();
  } a;

  int &test1() {
    a.nofail();
  } // expected-warning {{control reaches end of non-void function}}

  int &test2() {
    a.fail();
  }
}

namespace destructor_tests {
  __attribute__((noreturn)) void fail();

  struct A {
    ~A() __attribute__((noreturn)) { fail(); }
  };
  struct B {
    B() {}
    ~B() __attribute__((noreturn)) { fail(); }
  };
  struct C : A {};
  struct D : B {};
  struct E : virtual A {};
  struct F : A, virtual B {};
  struct G : E {};
  struct H : virtual D {};
  struct I : A {};
  struct J : I {};
  struct K : virtual A {};
  struct L : K {};
  struct M : virtual C {};
  struct N : M {};
  struct O { N n; };

  __attribute__((noreturn)) void test_1() { A a; }
  __attribute__((noreturn)) void test_2() { B b; }
  __attribute__((noreturn)) void test_3() { C c; }
  __attribute__((noreturn)) void test_4() { D d; }
  __attribute__((noreturn)) void test_5() { E e; }
  __attribute__((noreturn)) void test_6() { F f; }
  __attribute__((noreturn)) void test_7() { G g; }
  __attribute__((noreturn)) void test_8() { H h; }
  __attribute__((noreturn)) void test_9() { I i; }
  __attribute__((noreturn)) void test_10() { J j; }
  __attribute__((noreturn)) void test_11() { K k; }
  __attribute__((noreturn)) void test_12() { L l; }
  __attribute__((noreturn)) void test_13() { M m; }
  __attribute__((noreturn)) void test_14() { N n; }
  __attribute__((noreturn)) void test_15() { O o; }

  __attribute__((noreturn)) void test_16() { const A& a = A(); }
  __attribute__((noreturn)) void test_17() { const B& b = B(); }
  __attribute__((noreturn)) void test_18() { const C& c = C(); }
  __attribute__((noreturn)) void test_19() { const D& d = D(); }
  __attribute__((noreturn)) void test_20() { const E& e = E(); }
  __attribute__((noreturn)) void test_21() { const F& f = F(); }
  __attribute__((noreturn)) void test_22() { const G& g = G(); }
  __attribute__((noreturn)) void test_23() { const H& h = H(); }
  __attribute__((noreturn)) void test_24() { const I& i = I(); }
  __attribute__((noreturn)) void test_25() { const J& j = J(); }
  __attribute__((noreturn)) void test_26() { const K& k = K(); }
  __attribute__((noreturn)) void test_27() { const L& l = L(); }
  __attribute__((noreturn)) void test_28() { const M& m = M(); }
  __attribute__((noreturn)) void test_29() { const N& n = N(); }
  __attribute__((noreturn)) void test_30() { const O& o = O(); }

  struct AA {};
  struct BB { BB() {} ~BB() {} };
  struct CC : AA {};
  struct DD : BB {};
  struct EE : virtual AA {};
  struct FF : AA, virtual BB {};
  struct GG : EE {};
  struct HH : virtual DD {};
  struct II : AA {};
  struct JJ : II {};
  struct KK : virtual AA {};
  struct LL : KK {};
  struct MM : virtual CC {};
  struct NN : MM {};
  struct OO { NN n; };

  __attribute__((noreturn)) void test_31() {
    AA a;
    BB b;
    CC c;
    DD d;
    EE e;
    FF f;
    GG g;
    HH h;
    II i;
    JJ j;
    KK k;
    LL l;
    MM m;
    NN n;
    OO o;

    const AA& aa = AA();
    const BB& bb = BB();
    const CC& cc = CC();
    const DD& dd = DD();
    const EE& ee = EE();
    const FF& ff = FF();
    const GG& gg = GG();
    const HH& hh = HH();
    const II& ii = II();
    const JJ& jj = JJ();
    const KK& kk = KK();
    const LL& ll = LL();
    const MM& mm = MM();
    const NN& nn = NN();
    const OO& oo = OO();
  }  // expected-warning {{function declared 'noreturn' should not return}}

  struct P {
    ~P() __attribute__((noreturn)) { fail(); }
    void foo() {}
  };
  struct Q : P { };
  __attribute__((noreturn)) void test31() {
    P().foo();
  }
  __attribute__((noreturn)) void test32() {
    Q().foo();
  }

  struct R {
    A a[5];
  };
  __attribute__((noreturn)) void test33() {
    R r;
  }

  // FIXME: Code flow analysis does not preserve information about non-null
  // pointers, so it can't determine that this function is noreturn.
  __attribute__((noreturn)) void test34() {
    A *a = new A;
    delete a;
  }  // expected-warning {{function declared 'noreturn' should not return}}

  struct S {
    virtual ~S();
  };
  struct T : S {
    __attribute__((noreturn)) ~T();
  };

  // FIXME: Code flow analysis does not preserve information about non-null
  // pointers or derived class pointers,  so it can't determine that this
  // function is noreturn.
  __attribute__((noreturn)) void test35() {
    S *s = new T;
    delete s;
  }  // expected-warning {{function declared 'noreturn' should not return}}
}

// PR5620
void f0() __attribute__((__noreturn__));
void f1(void (*)());
void f2() { f1(f0); }

// Taking the address of a noreturn function
void test_f0a() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// Taking the address of an overloaded noreturn function 
void f0(int) __attribute__((__noreturn__));

void test_f0b() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// No-returned function pointers
typedef void (* noreturn_fp)() __attribute__((noreturn));

void f3(noreturn_fp); // expected-note{{candidate function}}

void test_f3() {
  f3(f0); // okay
  f3(f2); // expected-error{{no matching function for call}}
}


class xpto {
  int blah() __attribute__((noreturn));
};

int xpto::blah() {
  return 3; // expected-warning {{function 'blah' declared 'noreturn' should not return}}
}

// PR12948

namespace PR12948 {
  template<int>
  void foo() __attribute__((__noreturn__));

  template<int>
  void foo() {
    while (1) continue;
  }

  void bar() __attribute__((__noreturn__));

  void bar() {
    foo<0>();
  }


  void baz() __attribute__((__noreturn__));
  typedef void voidfn();
  voidfn baz;

  template<typename> void wibble()  __attribute__((__noreturn__));
  template<typename> voidfn wibble;
}

// PR15291
// Overload resolution per over.over should allow implicit noreturn adjustment.
namespace PR15291 {
  __attribute__((noreturn)) void foo(int) {}
  __attribute__((noreturn)) void foo(double) {}

  template <typename T>
  __attribute__((noreturn)) void bar(T) {}

  void baz(int) {}
  void baz(double) {}

  template <typename T>
  void qux(T) {}

  // expected-note@+5 {{candidate function template not viable: no overload of 'baz' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+4 {{candidate function template not viable: no overload of 'qux' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+3 {{candidate function template not viable: no overload of 'bar' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+2 {{candidate function template not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  // expected-note@+1 {{candidate function template not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  template <typename T> void accept_T(T) {}

  // expected-note@+1 {{candidate function not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  void accept_fptr(void (*f)(int)) {
    f(42);
  }

  // expected-note@+2 {{candidate function not viable: no overload of 'baz' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+1 {{candidate function not viable: no overload of 'qux' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  void accept_noreturn_fptr(void __attribute__((noreturn)) (*f)(int)) {
    f(42);
  }

  typedef void (*fptr_t)(int);
  typedef void __attribute__((noreturn)) (*fptr_noreturn_t)(int);

  // expected-note@+1 {{candidate function not viable: no overload of 'bar' matching 'PR15291::fptr_t' (aka 'void (*)(int)') for 1st argument}}
  void accept_fptr_t(fptr_t f) {
    f(42);
  }

  // expected-note@+2 {{candidate function not viable: no overload of 'baz' matching 'PR15291::fptr_noreturn_t' (aka 'void (*)(int) __attribute__((noreturn))') for 1st argument}}
  // expected-note@+1 {{candidate function not viable: no overload of 'qux' matching 'PR15291::fptr_noreturn_t' (aka 'void (*)(int) __attribute__((noreturn))') for 1st argument}}
  void accept_fptr_noreturn_t(fptr_noreturn_t f) {
    f(42);
  }

  // Stripping noreturn should work if everything else is correct.
  void strip_noreturn() {
    accept_fptr(foo);
    accept_fptr(bar<int>);
    accept_fptr(bar<double>); // expected-error {{no matching function for call to 'accept_fptr'}}

    accept_fptr_t(foo);
    accept_fptr_t(bar<int>);
    accept_fptr_t(bar<double>); // expected-error {{no matching function for call to 'accept_fptr_t'}}

    accept_T<void __attribute__((noreturn)) (*)(int)>(foo);
    accept_T<void __attribute__((noreturn)) (*)(int)>(bar<int>);
    accept_T<void __attribute__((noreturn)) (*)(int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}

    accept_T<void (*)(int)>(foo);
    accept_T<void (*)(int)>(bar<int>);
    accept_T<void (*)(int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}

    accept_T<void (int)>(foo);
    accept_T<void (int)>(bar<int>);
    accept_T<void (int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}
  }

  // Introducing noreturn should not work.
  void introduce_noreturn() {
    accept_noreturn_fptr(baz); // expected-error {{no matching function for call to 'accept_noreturn_fptr'}}
    accept_noreturn_fptr(qux<int>); // expected-error {{no matching function for call to 'accept_noreturn_fptr'}}

    accept_fptr_noreturn_t(baz); // expected-error {{no matching function for call to 'accept_fptr_noreturn_t'}}
    accept_fptr_noreturn_t(qux<int>); // expected-error {{no matching function for call to 'accept_fptr_noreturn_t'}}

    accept_T<void __attribute__((noreturn)) (*)(int)>(baz); // expected-error {{no matching function for call to 'accept_T'}}
    accept_T<void __attribute__((noreturn)) (*)(int)>(qux<int>); // expected-error {{no matching function for call to 'accept_T'}}
  }
}
