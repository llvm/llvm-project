// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// PR13819 -- __SIZE_TYPE__ is incompatible.
typedef __SIZE_TYPE__ size_t; // expected-error 0-1 {{extension}}

#if __cplusplus < 201103L
#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define fold
#endif

namespace dr200 { // dr200: dup 214
  template <class T> T f(int);
  template <class T, class U> T f(U) = delete; // expected-error 0-1{{extension}}

  void g() {
    f<int>(1);
  }
}

// dr201 FIXME: write codegen test

namespace dr202 { // dr202: yes
  template<typename T> T f();
  template<int (*g)()> struct X {
    int arr[fold(g == &f<int>) ? 1 : -1];
  };
  template struct X<f>;
}

// FIXME (export) dr204: no

namespace dr206 { // dr206: yes
  struct S; // expected-note 2{{declaration}}
  template<typename T> struct Q { S s; }; // expected-error {{incomplete}}
  template<typename T> void f() { S s; } // expected-error {{incomplete}}
}

namespace dr207 { // dr207: yes
  class A {
  protected:
    static void f() {}
  };
  class B : A {
  public:
    using A::f;
    void g() {
      A::f();
      f();
    }
  };
}

// dr208 FIXME: write codegen test

namespace dr209 { // dr209: yes
  class A {
    void f(); // expected-note {{here}}
  };
  class B {
    friend void A::f(); // expected-error {{private}}
  };
}

// dr210 FIXME: write codegen test

namespace dr211 { // dr211: yes
  struct A {
    A() try {
      throw 0;
    } catch (...) {
      return; // expected-error {{return in the catch of a function try block of a constructor}}
    }
  };
}

namespace dr213 { // dr213: yes
  template <class T> struct A : T {
    void h(T t) {
      char &r1 = f(t);
      int &r2 = g(t); // expected-error {{undeclared}}
    }
  };
  struct B {
    int &f(B);
    int &g(B); // expected-note {{in dependent base class}}
  };
  char &f(B);

  template void A<B>::h(B); // expected-note {{instantiation}}
}

namespace dr214 { // dr214: yes
  template<typename T, typename U> T checked_cast(U from) { U::error; }
  template<typename T, typename U> T checked_cast(U *from);
  class C {};
  void foo(int *arg) { checked_cast<const C *>(arg); }

  template<typename T> T f(int);
  template<typename T, typename U> T f(U) { T::error; }
  void g() {
    f<int>(1);
  }
}

namespace dr215 { // dr215: yes
  template<typename T> class X {
    friend void T::foo();
    int n;
  };
  struct Y {
    void foo() { (void)+X<Y>().n; }
  };
}

namespace dr216 { // dr216: no
  // FIXME: Should reject this: 'f' has linkage but its type does not,
  // and 'f' is odr-used but not defined in this TU.
  typedef enum { e } *E;
  void f(E);
  void g(E e) { f(e); }

  struct S {
    // FIXME: Should reject this: 'f' has linkage but its type does not,
    // and 'f' is odr-used but not defined in this TU.
    typedef enum { e } *E;
    void f(E);
  };
  void g(S s, S::E e) { s.f(e); }
}

namespace dr217 { // dr217: yes
  template<typename T> struct S {
    void f(int);
  };
  template<typename T> void S<T>::f(int = 0) {} // expected-error {{default arguments cannot be added}}
}

namespace dr218 { // dr218: yes
  namespace A {
    struct S {};
    void f(S);
  }
  namespace B {
    struct S {};
    void f(S);
  }

  struct C {
    int f;
    void test1(A::S as) { f(as); } // expected-error {{called object type 'int'}}
    void test2(A::S as) { void f(); f(as); } // expected-error {{too many arguments}} expected-note {{}}
    void test3(A::S as) { using A::f; f(as); } // ok
    void test4(A::S as) { using B::f; f(as); } // ok
    void test5(A::S as) { int f; f(as); } // expected-error {{called object type 'int'}}
    void test6(A::S as) { struct f {}; (void) f(as); } // expected-error {{no matching conversion}} expected-note +{{}}
  };

  namespace D {
    struct S {};
    struct X { void operator()(S); } f;
  }
  void testD(D::S ds) { f(ds); } // expected-error {{undeclared identifier}}

  namespace E {
    struct S {};
    struct f { f(S); };
  }
  void testE(E::S es) { f(es); } // expected-error {{undeclared identifier}}

  namespace F {
    struct S {
      template<typename T> friend void f(S, T) {}
    };
  }
  void testF(F::S fs) { f(fs, 0); }

  namespace G {
    namespace X {
      int f;
      struct A {};
    }
    namespace Y {
      template<typename T> void f(T);
      struct B {};
    }
    template<typename A, typename B> struct C {};
  }
  void testG(G::C<G::X::A, G::Y::B> gc) { f(gc); }
}

// dr219: na
// dr220: na

namespace dr221 { // dr221: yes
  struct A { // expected-note 2-4{{candidate}}
    A &operator=(int&); // expected-note 2{{candidate}}
    A &operator+=(int&);
    static A &operator=(A&, double&); // expected-error {{cannot be a static member}}
    static A &operator+=(A&, double&); // expected-error {{cannot be a static member}}
    friend A &operator=(A&, char&); // expected-error {{must be a non-static member function}}
    friend A &operator+=(A&, char&);
  };
  A &operator=(A&, float&); // expected-error {{must be a non-static member function}}
  A &operator+=(A&, float&);

  void test(A a, int n, char c, float f) {
    a = n;
    a += n;
    a = c; // expected-error {{no viable}}
    a += c;
    a = f; // expected-error {{no viable}}
    a += f;
  }
}

namespace dr222 { // dr222: dup 637
  void f(int a, int b, int c, int *x) {
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsequenced"
    void((a += b) += c);
    void((a += b) + (a += c)); // expected-warning {{multiple unsequenced modifications to 'a'}}

    x[a++] = a; // expected-warning {{unsequenced modification and access to 'a'}}

    a = b = 0; // ok, read and write of 'b' are sequenced

    a = (b = a++); // expected-warning {{multiple unsequenced modifications to 'a'}}
    a = (b = ++a);
#pragma clang diagnostic pop
  }
}

// dr223: na

namespace dr224 { // dr224: no
  namespace example1 {
    template <class T> class A {
      typedef int type;
      A::type a;
      A<T>::type b;
      A<T*>::type c; // expected-error {{missing 'typename'}}
      ::dr224::example1::A<T>::type d;

      class B {
        typedef int type;

        A::type a;
        A<T>::type b;
        A<T*>::type c; // expected-error {{missing 'typename'}}
        ::dr224::example1::A<T>::type d;

        B::type e;
        A<T>::B::type f;
        A<T*>::B::type g; // expected-error {{missing 'typename'}}
        typename A<T*>::B::type h;
      };
    };

    template <class T> class A<T*> {
      typedef int type;
      A<T*>::type a;
      A<T>::type b; // expected-error {{missing 'typename'}}
    };

    template <class T1, class T2, int I> struct B {
      typedef int type;
      B<T1, T2, I>::type b1;
      B<T2, T1, I>::type b2; // expected-error {{missing 'typename'}}

      typedef T1 my_T1;
      static const int my_I = I;
      static const int my_I2 = I+0;
      static const int my_I3 = my_I;
      B<my_T1, T2, my_I>::type b3; // FIXME: expected-error {{missing 'typename'}}
      B<my_T1, T2, my_I2>::type b4; // expected-error {{missing 'typename'}}
      B<my_T1, T2, my_I3>::type b5; // FIXME: expected-error {{missing 'typename'}}
    };
  }

  namespace example2 {
    template <int, typename T> struct X { typedef T type; };
    template <class T> class A {
      static const int i = 5;
      X<i, int>::type w; // FIXME: expected-error {{missing 'typename'}}
      X<A::i, char>::type x; // FIXME: expected-error {{missing 'typename'}}
      X<A<T>::i, double>::type y; // FIXME: expected-error {{missing 'typename'}}
      X<A<T*>::i, long>::type z; // expected-error {{missing 'typename'}}
      int f();
    };
    template <class T> int A<T>::f() {
      return i;
    }
  }
}

// dr225: yes
template<typename T> void dr225_f(T t) { dr225_g(t); } // expected-error {{call to function 'dr225_g' that is neither visible in the template definition nor found by argument-dependent lookup}}
void dr225_g(int); // expected-note {{should be declared prior to the call site}}
template void dr225_f(int); // expected-note {{in instantiation of}}

namespace dr226 { // dr226: no
  template<typename T = void> void f() {}
#if __cplusplus < 201103L
  // expected-error@-2 {{extension}}
  // FIXME: This appears to be wrong: default arguments for function templates
  // are listed as a defect (in c++98) not an extension. EDG accepts them in
  // strict c++98 mode.
#endif
  template<typename T> struct S {
    template<typename U = void> void g();
#if __cplusplus < 201103L
  // expected-error@-2 {{extension}}
#endif
    template<typename U> struct X;
    template<typename U> void h();
  };
  template<typename T> template<typename U> void S<T>::g() {}
  template<typename T> template<typename U = void> struct S<T>::X {}; // expected-error {{cannot add a default template arg}}
  template<typename T> template<typename U = void> void S<T>::h() {} // expected-error {{cannot add a default template arg}}

  template<typename> void friend_h();
  struct A {
    // FIXME: This is ill-formed.
    template<typename=void> struct friend_B;
    // FIXME: f, h, and i are ill-formed.
    //  f is ill-formed because it is not a definition.
    //  h and i are ill-formed because they are not the only declarations of the
    //  function in the translation unit.
    template<typename=void> void friend_f();
    template<typename=void> void friend_g() {}
    template<typename=void> void friend_h() {}
    template<typename=void> void friend_i() {}
#if __cplusplus < 201103L
  // expected-error@-5 {{extension}} expected-error@-4 {{extension}}
  // expected-error@-4 {{extension}} expected-error@-3 {{extension}}
#endif
  };
  template<typename> void friend_i();

  template<typename=void, typename X> void foo(X) {}
  template<typename=void, typename X> struct Foo {}; // expected-error {{missing a default argument}} expected-note {{here}}
#if __cplusplus < 201103L
  // expected-error@-3 {{extension}}
#endif

  template<typename=void, typename X, typename, typename Y> int foo(X, Y);
  template<typename, typename X, typename=void, typename Y> int foo(X, Y);
  int x = foo(0, 0);
#if __cplusplus < 201103L
  // expected-error@-4 {{extension}}
  // expected-error@-4 {{extension}}
#endif
}

void dr227(bool b) { // dr227: yes
  if (b)
    int n;
  else
    int n;
}

namespace dr228 { // dr228: yes
  template <class T> struct X {
    void f();
  };
  template <class T> struct Y {
    void g(X<T> x) { x.template X<T>::f(); }
  };
}

namespace dr229 { // dr229: yes
  template<typename T> void f();
  template<typename T> void f<T*>() {} // expected-error {{function template partial specialization}}
  template<> void f<int>() {}
}

namespace dr230 { // dr230: yes
  struct S {
    S() { f(); } // expected-warning {{call to pure virtual member function}}
    virtual void f() = 0; // expected-note {{declared here}}
  };
}

namespace dr231 { // dr231: yes
  namespace outer {
    namespace inner {
      int i; // expected-note {{here}}
    }
    void f() { using namespace inner; }
    int j = i; // expected-error {{undeclared identifier 'i'; did you mean 'inner::i'?}}
  }
}

// dr234: na
// dr235: na

namespace dr236 { // dr236: yes
  void *p = int();
#if __cplusplus < 201103L
  // expected-warning@-2 {{null pointer}}
#else
  // expected-error@-4 {{cannot initialize}}
#endif
}

namespace dr237 { // dr237: dup 470
  template<typename T> struct A { void f() { T::error; } };
  template<typename T> struct B : A<T> {};
  template struct B<int>; // ok
}

namespace dr239 { // dr239: yes
  namespace NS {
    class T {};
    void f(T);
    float &g(T, int);
  }
  NS::T parm;
  int &g(NS::T, float);
  int main() {
    f(parm);
    float &r = g(parm, 1);
    extern int &g(NS::T, float);
    int &s = g(parm, 1);
  }
}

// dr240: dup 616

namespace dr241 { // dr241: yes
  namespace A {
    struct B {};
    template <int X> void f(); // expected-note 3{{candidate}}
    template <int X> void g(B);
  }
  namespace C {
    template <class T> void f(T t); // expected-note 2{{candidate}}
    template <class T> void g(T t); // expected-note {{candidate}}
  }
  void h(A::B b) {
    f<3>(b); // expected-error 0-1{{C++2a extension}} expected-error {{no matching}}
    g<3>(b); // expected-error 0-1{{C++2a extension}}
    A::f<3>(b); // expected-error {{no matching}}
    A::g<3>(b);
    C::f<3>(b); // expected-error {{no matching}}
    C::g<3>(b); // expected-error {{no matching}}
    using C::f;
    using C::g;
    f<3>(b); // expected-error {{no matching}}
    g<3>(b);
  }
}

namespace dr243 { // dr243: yes
  struct B;
  struct A {
    A(B); // expected-note {{candidate}}
  };
  struct B {
    operator A() = delete; // expected-error 0-1{{extension}} expected-note {{candidate}}
  } b;
  A a1(b);
  A a2 = b; // expected-error {{ambiguous}}
}

namespace dr244 { // dr244: partial
  struct B {}; struct D : B {}; // expected-note {{here}}

  D D_object;
  typedef B B_alias;
  B* B_ptr = &D_object;

  void f() {
    D_object.~B(); // expected-error {{expression does not match the type}}
    D_object.B::~B();
    B_ptr->~B();
    B_ptr->~B_alias();
    B_ptr->B_alias::~B();
    // This is valid under DR244.
    B_ptr->B_alias::~B_alias();
    B_ptr->dr244::~B(); // expected-error {{refers to a member in namespace}}
    B_ptr->dr244::~B_alias(); // expected-error {{refers to a member in namespace}}
  }

  namespace N {
    template<typename T> struct E {};
    typedef E<int> F;
  }
  void g(N::F f) {
    typedef N::F G;
    f.~G();
    f.G::~E();
    f.G::~F(); // expected-error {{expected the class name after '~' to name a destructor}}
    f.G::~G();
    // This is technically ill-formed; E is looked up in 'N::' and names the
    // class template, not the injected-class-name of the class. But that's
    // probably a bug in the standard.
    f.N::F::~E();
    // This is valid; we look up the second F in the same scope in which we
    // found the first one, that is, 'N::'.
    f.N::F::~F(); // FIXME: expected-error {{expected the class name after '~' to name a destructor}}
    // This is technically ill-formed; G is looked up in 'N::' and is not found;
    // as above, this is probably a bug in the standard.
    f.N::F::~G();
  }
}

namespace dr245 { // dr245: yes
  struct S {
    enum E {}; // expected-note {{here}}
    class E *p; // expected-error {{does not match previous declaration}}
  };
}

namespace dr246 { // dr246: yes
  struct S {
    S() try { // expected-note {{try block}}
      throw 0;
X: ;
    } catch (int) {
      goto X; // expected-error {{cannot jump}}
    }
  };
}

namespace dr247 { // dr247: yes
  struct A {};
  struct B : A {
    void f();
    void f(int);
  };
  void (A::*f)() = (void (A::*)())&B::f;

  struct C {
    void f();
    void f(int);
  };
  struct D : C {};
  void (C::*g)() = &D::f;
  void (D::*h)() = &D::f;

  struct E {
    void f();
  };
  struct F : E {
    using E::f;
    void f(int);
  };
  void (F::*i)() = &F::f;
}

namespace dr248 { // dr248: yes c++11
  // FIXME: Should this also apply to c++98 mode? This was a DR against C++98.
  int \u040d\u040e = 0;
#if __cplusplus < 201103L
  // FIXME: expected-error@-2 {{expected ';'}}
#endif
}

namespace dr249 { // dr249: yes
  template<typename T> struct X { void f(); };
  template<typename T> void X<T>::f() {}
}

namespace dr250 { // dr250: yes
  typedef void (*FPtr)(double x[]);

  template<int I> void f(double x[]);
  FPtr fp = &f<3>;

  template<int I = 3> void g(double x[]); // expected-error 0-1{{extension}}
  FPtr gp = &g<>;
}

namespace dr252 { // dr252: yes
  struct A {
    void operator delete(void*); // expected-note {{found}}
  };
  struct B {
    void operator delete(void*); // expected-note {{found}}
  };
  struct C : A, B {
    virtual ~C();
  };
  C::~C() {} // expected-error {{'operator delete' found in multiple base classes}}

  struct D {
    void operator delete(void*, int); // expected-note {{here}}
    virtual ~D();
  };
  D::~D() {} // expected-error {{no suitable member 'operator delete'}}

  struct E {
    void operator delete(void*, int);
    void operator delete(void*) = delete; // expected-error 0-1{{extension}} expected-note 1-2 {{here}}
    virtual ~E(); // expected-error 0-1 {{attempt to use a deleted function}}
  };
  E::~E() {} // expected-error {{attempt to use a deleted function}}

  struct F {
    // If both functions are available, the first one is a placement delete.
    void operator delete(void*, size_t);
    void operator delete(void*) = delete; // expected-error 0-1{{extension}} expected-note {{here}}
    virtual ~F();
  };
  F::~F() {} // expected-error {{attempt to use a deleted function}}

  struct G {
    void operator delete(void*, size_t);
    virtual ~G();
  };
  G::~G() {}
}

namespace dr254 { // dr254: yes
  template<typename T> struct A {
    typedef typename T::type type; // ok even if this is a typedef-name, because
                                   // it's not an elaborated-type-specifier
    typedef struct T::type foo; // expected-error {{typedef 'type' cannot be referenced with a struct specifier}}
  };
  struct B { struct type {}; };
  struct C { typedef struct {} type; }; // expected-note {{here}}
  A<B>::type n;
  A<C>::type n; // expected-note {{instantiation of}}
}

// dr256: dup 624

namespace dr257 { // dr257: yes
  struct A { A(int); }; // expected-note {{here}}
  struct B : virtual A {
    B() {}
    virtual void f() = 0;
  };
  struct C : B {
    C() {}
  };
  struct D : B {
    D() {} // expected-error {{must explicitly initialize the base class 'dr257::A'}}
    void f();
  };
}

namespace dr258 { // dr258: yes
  struct A {
    void f(const int);
    template<typename> void g(int);
    float &h() const;
  };
  struct B : A {
    using A::f;
    using A::g;
    using A::h;
    int &f(int);
    template<int> int &g(int); // expected-note {{candidate}}
    int &h();
  } b;
  int &w = b.f(0);
  int &x = b.g<int>(0); // expected-error {{no match}}
  int &y = b.h();
  float &z = const_cast<const B&>(b).h();

  struct C {
    virtual void f(const int) = 0;
  };
  struct D : C {
    void f(int);
  } d;

  struct E {
    virtual void f() = 0; // expected-note {{unimplemented}}
  };
  struct F : E {
    void f() const {}
  } f; // expected-error {{abstract}}
}

namespace dr259 { // dr259: 4
  template<typename T> struct A {};
  template struct A<int>; // expected-note {{previous}}
  template struct A<int>; // expected-error {{duplicate explicit instantiation}}

  template<> struct A<float>; // expected-note {{previous}}
  template struct A<float>; // expected-warning {{has no effect}}

  template struct A<char>; // expected-note {{here}}
  template<> struct A<char>; // expected-error {{explicit specialization of 'dr259::A<char>' after instantiation}}

  template<> struct A<double>;
  template<> struct A<double>;
  template<> struct A<double> {}; // expected-note {{here}}
  template<> struct A<double> {}; // expected-error {{redefinition}}

  template<typename T> struct B; // expected-note {{here}}
  template struct B<int>; // expected-error {{undefined}}

  template<> struct B<float>; // expected-note {{previous}}
  template struct B<float>; // expected-warning {{has no effect}}
}

// FIXME: When dr260 is resolved, also add tests for DR507.

namespace dr261 { // dr261: no
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wused-but-marked-unused"

  // FIXME: This is ill-formed, with a diagnostic required, because operator new
  // and operator delete are inline and odr-used, but not defined in this
  // translation unit.
  // We're also missing the -Wused-but-marked-unused diagnostic here.
  struct A {
    inline void *operator new(size_t) __attribute__((unused));
    inline void operator delete(void*) __attribute__((unused));
    A() {}
  };

  // FIXME: This is ill-formed, with a required diagnostic, for the same
  // reason.
  struct B {
    inline void operator delete(void*) __attribute__((unused));
    ~B() {}
  };
  struct C {
    inline void operator delete(void*) __attribute__((unused));
    virtual ~C() {} // expected-warning {{'operator delete' was marked unused but was used}}
  };

  struct D {
    inline void operator delete(void*) __attribute__((unused));
  };
  void h() { C::operator delete(0); } // expected-warning {{marked unused but was used}}

#pragma clang diagnostic pop
}

namespace dr262 { // dr262: yes
  int f(int = 0, ...);
  int k = f();
  int l = f(0);
  int m = f(0, 0);
}

namespace dr263 { // dr263: yes
  struct X {};
  struct Y {
#if __cplusplus < 201103L
    friend X::X() throw();
    friend X::~X() throw();
#else
    friend constexpr X::X() noexcept;
    friend X::~X();
#endif
    Y::Y(); // expected-error {{extra qualification}}
    Y::~Y(); // expected-error {{extra qualification}}
  };
}

// dr265: dup 353
// dr266: na
// dr269: na
// dr270: na

namespace dr272 { // dr272: yes
  struct X {
    void f() {
      this->~X();
      X::~X();
      ~X(); // expected-error {{unary expression}}
    }
  };
}

#include <stdarg.h>
#include <stddef.h>
namespace dr273 { // dr273: yes
  struct A {
    int n;
  };
  void operator&(A);
  void f(A a, ...) {
    offsetof(A, n);
    va_list val;
    va_start(val, a);
    va_end(val);
  }
}

// dr274: na

namespace dr275 { // dr275: no
  namespace N {
    template <class T> void f(T) {} // expected-note 1-4{{here}}
    template <class T> void g(T) {} // expected-note {{candidate}}
    template <> void f(int);
    template <> void f(char);
    template <> void f(double);
    template <> void g(char);
  }

  using namespace N;

  namespace M {
    template <> void N::f(char) {} // expected-error {{'M' does not enclose namespace 'N'}}
    template <class T> void g(T) {}
    template <> void g(char) {}
    template void f(long);
#if __cplusplus >= 201103L
    // FIXME: this should be rejected in c++98 too
    // expected-error@-3 {{must occur in namespace 'N'}}
#endif
    template void N::f(unsigned long);
#if __cplusplus >= 201103L
    // FIXME: this should be rejected in c++98 too
    // expected-error@-3 {{not in a namespace enclosing 'N'}}
#endif
    template void h(long); // expected-error {{does not refer to a function template}}
    template <> void f(double) {} // expected-error {{no function template matches}}
  }

  template <class T> void g(T) {} // expected-note {{candidate}}

  template <> void N::f(char) {}
  template <> void f(int) {} // expected-error {{no function template matches}}

  template void f(short);
#if __cplusplus >= 201103L
  // FIXME: this should be rejected in c++98 too
  // expected-error@-3 {{must occur in namespace 'N'}}
#endif
  template void N::f(unsigned short);

  // FIXME: this should probably be valid. the wording from the issue
  // doesn't clarify this, but it follows from the usual rules.
  template void g(int); // expected-error {{ambiguous}}

  // FIXME: likewise, this should also be valid.
  template<typename T> void f(T) {} // expected-note {{candidate}}
  template void f(short); // expected-error {{ambiguous}}
}

// dr276: na

namespace dr277 { // dr277: yes
  typedef int *intp;
  int *p = intp();
  int a[fold(intp() ? -1 : 1)];
}

namespace dr280 { // dr280: yes
  typedef void f0();
  typedef void f1(int);
  typedef void f2(int, int);
  typedef void f3(int, int, int);
  struct A {
    operator f1*(); // expected-note {{here}} expected-note {{candidate}}
    operator f2*();
  };
  struct B {
    operator f0*(); // expected-note {{candidate}}
  private:
    operator f3*(); // expected-note {{here}} expected-note {{candidate}}
  };
  struct C {
    operator f0*(); // expected-note {{candidate}}
    operator f1*(); // expected-note {{candidate}}
    operator f2*(); // expected-note {{candidate}}
    operator f3*(); // expected-note {{candidate}}
  };
  struct D : private A, B { // expected-note 2{{here}}
    operator f2*(); // expected-note {{candidate}}
  } d;
  struct E : C, D {} e;
  void g() {
    d(); // ok, public
    d(0); // expected-error {{private member of 'dr280::A'}} expected-error {{private base class 'dr280::A'}}
    d(0, 0); // ok, suppressed by member in D
    d(0, 0, 0); // expected-error {{private member of 'dr280::B'}}
    e(); // expected-error {{ambiguous}}
    e(0); // expected-error {{ambiguous}}
    e(0, 0); // expected-error {{ambiguous}}
    e(0, 0, 0); // expected-error {{ambiguous}}
  }
}

namespace dr281 { // dr281: no
  void a();
  inline void b();

  void d();
  inline void e();

  struct S {
    friend inline void a(); // FIXME: ill-formed
    friend inline void b();
    friend inline void c(); // FIXME: ill-formed
    friend inline void d() {}
    friend inline void e() {}
    friend inline void f() {}
  };
}

namespace dr283 { // dr283: yes
  template<typename T> // expected-note 2{{here}}
  struct S {
    friend class T; // expected-error {{shadows}}
    class T; // expected-error {{shadows}}
  };
}

namespace dr284 { // dr284: no
  namespace A {
    struct X;
    enum Y {};
    class Z {};
  }
  namespace B {
    struct W;
    using A::X;
    using A::Y;
    using A::Z;
  }
  struct B::V {}; // expected-error {{no struct named 'V'}}
  struct B::W {};
  struct B::X {}; // FIXME: ill-formed
  enum B::Y e; // ok per dr417
  class B::Z z; // ok per dr417

  struct C {
    struct X;
    enum Y {};
    class Z {};
  };
  struct D : C {
    struct W;
    using C::X;
    using C::Y;
    using C::Z;
  };
  struct D::V {}; // expected-error {{no struct named 'V'}}
  struct D::W {};
  struct D::X {}; // FIXME: ill-formed
  enum D::Y e2; // ok per dr417
  class D::Z z2; // ok per dr417
}

namespace dr285 { // dr285: yes
  template<typename T> void f(T, int); // expected-note {{match}}
  template<typename T> void f(int, T); // expected-note {{match}}
  template<> void f<int>(int, int) {} // expected-error {{ambiguous}}
}

namespace dr286 { // dr286: yes
  template<class T> struct A {
    class C {
      template<class T2> struct B {}; // expected-note {{here}}
    };
  };

  template<class T>
  template<class T2>
  struct A<T>::C::B<T2*> { };

  A<short>::C::B<int*> absip; // expected-error {{private}}
}

// dr288: na

namespace dr289 { // dr289: yes
  struct A; // expected-note {{forward}}
  struct B : A {}; // expected-error {{incomplete}}

  template<typename T> struct C { typename T::error error; }; // expected-error {{cannot be used prior to '::'}}
  struct D : C<int> {}; // expected-note {{instantiation}}
}

// dr290: na
// dr291: dup 391
// dr292 FIXME: write a codegen test

namespace dr294 { // dr294: no
  void f() throw(int);
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++17 does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
  int main() {
    (void)static_cast<void (*)() throw()>(f); // FIXME: ill-formed in C++14 and before
#if __cplusplus > 201402L
    // FIXME: expected-error@-2 {{not allowed}}
    //
    // Irony: the above is valid in C++17 and beyond, but that's exactly when
    // we reject it. In C++14 and before, this is ill-formed because an
    // exception-specification is not permitted in a type-id. In C++17, this is
    // valid because it's the inverse of a standard conversion sequence
    // containing a function pointer conversion. (Well, it's actually not valid
    // yet, as a static_cast is not permitted to reverse a function pointer
    // conversion, but that is being changed by core issue).
#endif
    (void)static_cast<void (*)() throw(int)>(f); // FIXME: ill-formed in C++14 and before
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++17 does not allow}} expected-note@-2 {{use 'noexcept}}
#endif

    void (*p)() throw() = f; // expected-error-re {{{{not superset|different exception specification}}}}
    void (*q)() throw(int) = f;
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++17 does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
  }
}

namespace dr295 { // dr295: 3.7
  typedef int f();
  const f g; // expected-warning {{'const' qualifier on function type 'dr295::f' (aka 'int ()') has no effect}}
  f &r = g;
  template<typename T> struct X {
    const T &f;
  };
  X<f> x = {g};

  typedef int U();
  typedef const U U; // expected-warning {{'const' qualifier on function type 'dr295::U' (aka 'int ()') has no effect}}

  typedef int (*V)();
  typedef volatile U *V; // expected-warning {{'volatile' qualifier on function type 'dr295::U' (aka 'int ()') has no effect}}
}

namespace dr296 { // dr296: yes
  struct A {
    static operator int() { return 0; } // expected-error {{static}}
  };
}

namespace dr298 { // dr298: yes
  struct A {
    typedef int type;
    A();
    ~A();
  };
  typedef A B; // expected-note {{here}}
  typedef const A C; // expected-note {{here}}

  A::type i1;
  B::type i2;
  C::type i3;

  struct A a;
  struct B b; // expected-error {{typedef 'B' cannot be referenced with a struct specifier}}
  struct C c; // expected-error {{typedef 'C' cannot be referenced with a struct specifier}}

  B::B() {} // expected-error {{requires a type specifier}}
  B::A() {} // ok
  C::~C() {} // expected-error {{destructor cannot be declared using a typedef 'dr298::C' (aka 'const dr298::A') of the class name}}

  typedef struct D E; // expected-note {{here}}
  struct E {}; // expected-error {{conflicts with typedef}}

  struct F {
    ~F();
  };
  typedef const F G;
  G::~F() {} // ok
}

namespace dr299 { // dr299: yes c++11
  struct S {
    operator int();
  };
  struct T {
    operator int(); // expected-note {{}}
    operator unsigned short(); // expected-note {{}}
  };
  // FIXME: should this apply to c++98 mode?
  int *p = new int[S()]; // expected-error 0-1{{extension}}
  int *q = new int[T()]; // expected-error {{ambiguous}}
}
