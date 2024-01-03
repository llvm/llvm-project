// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98,cxx98-11,cxx98-14,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11,cxx98-11,cxx98-14,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11,since-cxx14,cxx98-14,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx14,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors

// FIXME: diagnostic above is emitted only on Windows platforms
// PR13819 -- __SIZE_TYPE__ is incompatible.
typedef __SIZE_TYPE__ size_t;
// cxx98-error@-1 0-1 {{'long long' is a C++11 extension}}

#if __cplusplus < 201103L
#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define fold
#endif

namespace dr200 { // dr200: dup 214
  template <class T> T f(int);
  template <class T, class U> T f(U) = delete;
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}

  void g() {
    f<int>(1);
  }
}

// dr201 FIXME: write codegen test

namespace dr202 { // dr202: 3.1
  template<typename T> T f();
  template<int (*g)()> struct X {
    int arr[fold(g == &f<int>) ? 1 : -1];
  };
  template struct X<f>;
}

// FIXME (export) dr204: no

namespace dr206 { // dr206: yes
  struct S; // #dr206-S
  template<typename T> struct Q { S s; };
  // expected-error@-1 {{field has incomplete type 'S'}}
  // expected-note@#dr206-S {{forward declaration of 'dr206::S'}}
  template<typename T> void f() { S s; }
  // expected-error@-1 {{variable has incomplete type 'S'}}
  // expected-note@#dr206-S {{forward declaration of 'dr206::S'}}
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

namespace dr209 { // dr209: 3.2
  class A {
    void f(); // #dr209-A-f
  };
  class B {
    friend void A::f();
    // expected-error@-1 {{friend function 'f' is a private member of 'dr209::A'}}
    // expected-note@#dr209-A-f {{implicitly declared private here}}
  };
}

// dr210 FIXME: write codegen test

namespace dr211 { // dr211: yes
  struct A {
    A() try {
      throw 0;
    } catch (...) {
      return;
      // expected-error@-1 {{return in the catch of a function try block of a constructor is illegal}}
    }
  };
}

namespace dr213 { // dr213: yes
  template <class T> struct A : T {
    void h(T t) {
      char &r1 = f(t);
      int &r2 = g(t);
      // expected-error@-1 {{explicit qualification required to use member 'g' from dependent base class}}
      // expected-note@#dr213-instantiation {{in instantiation of member function 'dr213::A<dr213::B>::h' requested here}}
      // expected-note@#dr213-B-g {{member is declared here}}
    }
  };
  struct B {
    int &f(B);
    int &g(B); // #dr213-B-g
  };
  char &f(B);

  template void A<B>::h(B); // #dr213-instantiation
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

namespace dr215 { // dr215: 2.9
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
  template<typename T> void S<T>::f(int = 0) {}
  // expected-error@-1 {{default arguments cannot be added to an out-of-line definition of a member of a class template}}
}

namespace dr218 { // dr218: yes
                  // NB: also dup 405
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
    void test1(A::S as) { f(as); }
    // expected-error@-1 {{called object type 'int' is not a function or function pointer}}
    void test2(A::S as) { void f(); f(as); }
    // expected-error@-1 {{too many arguments to function call, expected 0, have 1}}
    // expected-note@-2 {{'f' declared here}}
    void test3(A::S as) { using A::f; f(as); } // ok
    void test4(A::S as) { using B::f; f(as); } // ok
    void test5(A::S as) { int f; f(as); }
    // expected-error@-1 {{called object type 'int' is not a function or function pointer}}
    void test6(A::S as) { struct f {}; (void) f(as); }
    // expected-error@-1 {{no matching conversion for functional-style cast from 'A::S' to 'f'}}
    // expected-note@-2 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'A::S' to 'const f' for 1st argument}}
    // since-cxx11-note@-3 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'A::S' to 'f' for 1st argument}}
    // expected-note@-4 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  };

  namespace D {
    struct S {};
    struct X { void operator()(S); } f;
  }
  void testD(D::S ds) { f(ds); }
  // expected-error@-1 {{use of undeclared identifier 'f'}}

  namespace E {
    struct S {};
    struct f { f(S); };
  }
  void testE(E::S es) { f(es); }
  // expected-error@-1 {{use of undeclared identifier 'f'}}

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

namespace dr221 { // dr221: 3.6
  struct A { // #dr221-S
    A &operator=(int&); // #dr221-S-copy-assign
    A &operator+=(int&);
    static A &operator=(A&, double&);
    // expected-error@-1 {{overloaded 'operator=' cannot be a static member function}}
    static A &operator+=(A&, double&);
    // expected-error@-1 {{overloaded 'operator+=' cannot be a static member function}}
    friend A &operator=(A&, char&);
    // expected-error@-1 {{overloaded 'operator=' must be a non-static member function}}
    friend A &operator+=(A&, char&);
  };
  A &operator=(A&, float&);
  // expected-error@-1 {{overloaded 'operator=' must be a non-static member function}}
  A &operator+=(A&, float&);

  void test(A a, int n, char c, float f) {
    a = n;
    a += n;
    a = c;
    // expected-error@-1 {{no viable overloaded '='}}
    // expected-note@#dr221-S-copy-assign {{candidate function not viable: no known conversion from 'char' to 'int &' for 1st argument}}
    // since-cxx11-note@#dr221-S {{candidate function (the implicit move assignment operator) not viable: no known conversion from 'char' to 'A' for 1st argument}}
    // expected-note@#dr221-S {{candidate function (the implicit copy assignment operator) not viable: no known conversion from 'char' to 'const A' for 1st argument}}
    a += c;
    a = f;
    // expected-error@-1 {{no viable overloaded '='}}
    // expected-note@#dr221-S-copy-assign {{candidate function not viable: no known conversion from 'float' to 'int &' for 1st argument}}
    // since-cxx11-note@#dr221-S {{candidate function (the implicit move assignment operator) not viable: no known conversion from 'float' to 'A' for 1st argument}}
    // expected-note@#dr221-S {{candidate function (the implicit copy assignment operator) not viable: no known conversion from 'float' to 'const A' for 1st argument}}
    a += f;
  }
}

namespace dr222 { // dr222: dup 637
  void f(int a, int b, int c, int *x) {
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsequenced"
    void((a += b) += c);
    void((a += b) + (a += c));
    // expected-warning@-1 {{multiple unsequenced modifications to 'a'}}

    x[a++] = a;
    // cxx98-14-warning@-1 {{unsequenced modification and access to 'a'}}

    a = b = 0; // ok, read and write of 'b' are sequenced

    a = (b = a++);
    // cxx98-14-warning@-1 {{multiple unsequenced modifications to 'a'}}
    a = (b = ++a);
#pragma clang diagnostic pop
  }
}

// dr223: na

namespace dr224 { // dr224: 16
  namespace example1 {
    template <class T> class A {
      typedef int type;
      A::type a;
      A<T>::type b;
      A<T*>::type c;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name A<T *>::type; implicit 'typename' is a C++20 extension}}
      ::dr224::example1::A<T>::type d;

      class B {
        typedef int type;

        A::type a;
        A<T>::type b;
        A<T*>::type c;
        // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name A<T *>::type; implicit 'typename' is a C++20 extension}}
        ::dr224::example1::A<T>::type d;

        B::type e;
        A<T>::B::type f;
        A<T*>::B::type g;
        // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name A<T *>::B::type; implicit 'typename' is a C++20 extension}}
        typename A<T*>::B::type h;
      };
    };

    template <class T> class A<T*> {
      typedef int type;
      A<T*>::type a;
      A<T>::type b;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name A<T>::type; implicit 'typename' is a C++20 extension}}
    };

    template <class T1, class T2, int I> struct B {
      typedef int type;
      B<T1, T2, I>::type b1;
      B<T2, T1, I>::type b2;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B<T2, T1, I>::type; implicit 'typename' is a C++20 extension}}

      typedef T1 my_T1;
      static const int my_I = I;
      static const int my_I2 = I+0;
      static const int my_I3 = my_I;
      B<my_T1, T2, my_I>::type b3;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B<my_T1, T2, my_I>::type; implicit 'typename' is a C++20 extension}}
      B<my_T1, T2, my_I2>::type b4;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B<my_T1, T2, my_I2>::type; implicit 'typename' is a C++20 extension}}
      B<my_T1, T2, my_I3>::type b5;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B<my_T1, T2, my_I3>::type; implicit 'typename' is a C++20 extension}}
    };
  }

  namespace example2 {
    template <int, typename T> struct X { typedef T type; };
    template <class T> class A {
      static const int i = 5;
      X<i, int>::type w;
      X<A::i, char>::type x;
      X<A<T>::i, double>::type y;
      X<A<T*>::i, long>::type z;
      // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name X<A<T *>::i, long>::type; implicit 'typename' is a C++20 extension}}
      int f();
    };
    template <class T> int A<T>::f() {
      return i;
    }
  }
}

// dr225: yes
template<typename T> void dr225_f(T t) { dr225_g(t); }
// expected-error@-1 {{call to function 'dr225_g' that is neither visible in the template definition nor found by argument-dependent lookup}}
// expected-note@#dr225-f {{in instantiation of function template specialization 'dr225_f<int>' requested here}}
// expected-note@#dr225-g {{'dr225_g' should be declared prior to the call site}}
void dr225_g(int); // #dr225-g
template void dr225_f(int); // #dr225-f

namespace dr226 { // dr226: no
  // FIXME: This appears to be wrong: default arguments for function templates
  // are listed as a defect (in c++98) not an extension. EDG accepts them in
  // strict c++98 mode.
  template<typename T = void> void f() {}
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T> struct S {
    template<typename U = void> void g();
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
    template<typename U> struct X;
    template<typename U> void h();
  };
  template<typename T> template<typename U> void S<T>::g() {}
  template<typename T> template<typename U = void> struct S<T>::X {};
  // expected-error@-1 {{cannot add a default template argument to the definition of a member of a class template}}
  template<typename T> template<typename U = void> void S<T>::h() {}
  // expected-error@-1 {{cannot add a default template argument to the definition of a member of a class template}}

  template<typename> void friend_h();
  struct A {
    // FIXME: This is ill-formed.
    template<typename=void> struct friend_B;
    // FIXME: f, h, and i are ill-formed.
    //  f is ill-formed because it is not a definition.
    //  h and i are ill-formed because they are not the only declarations of the
    //  function in the translation unit.
    template<typename=void> void friend_f();
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
    template<typename=void> void friend_g() {}
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
    template<typename=void> void friend_h() {}
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
    template<typename=void> void friend_i() {}
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  };
  template<typename> void friend_i();

  template<typename=void, typename X> void foo(X) {}
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename=void, typename X> struct Foo {};
  // expected-error@-1 {{template parameter missing a default argument}}
  // expected-note@-2 {{previous default template argument defined here}}

  template<typename=void, typename X, typename, typename Y> int foo(X, Y);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename, typename X, typename=void, typename Y> int foo(X, Y);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  int x = foo(0, 0);
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

namespace dr229 { // dr229: 2.9
  template<typename T> void f();
  template<typename T> void f<T*>() {}
  // expected-error@-1 {{function template partial specialization is not allowed}}
  template<> void f<int>() {}
}

namespace dr230 { // dr230: 3.0
  struct S {
    S() { f(); }
    // expected-warning@-1 {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'dr230::S'}}
    // expected-note@#dr230-f {{'f' declared here}}
    virtual void f() = 0; // #dr230-f
  };
}

namespace dr231 { // dr231: yes
  namespace outer {
    namespace inner {
      int i; // #dr231-i
    }
    void f() { using namespace inner; }
    int j = i;
    // expected-error@-1 {{use of undeclared identifier 'i'; did you mean 'inner::i'?}}
    // expected-note@#dr231-i {{'inner::i' declared here}}
  }
}

// dr234: na
// dr235: na

namespace dr236 { // dr236: 3.2
  void *p = int();
  // cxx98-warning@-1 {{expression which evaluates to zero treated as a null pointer constant of type 'void *'}}
  // since-cxx11-error@-2 {{cannot initialize a variable of type 'void *' with an rvalue of type 'int'}}
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
    template <int X> void f(); // #dr241-A-f
    template <int X> void g(B);
  }
  namespace C {
    template <class T> void f(T t); // #dr241-C-f
    template <class T> void g(T t); // #dr241-C-g
  }
  void h(A::B b) {
    f<3>(b);
    // expected-error@-1 {{no matching function for call to 'f'}}
    //   expected-note@#dr241-A-f {{candidate function template not viable: requires 0 arguments, but 1 was provided}}
    // cxx98-17-error@-3 {{use of function template name with no prior declaration in function call with explicit template arguments is a C++20 extension}}
    g<3>(b);
    // cxx98-17-error@-1 {{use of function template name with no prior declaration in function call with explicit template arguments is a C++20 extension}}
    A::f<3>(b);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr241-A-f {{candidate function template not viable: requires 0 arguments, but 1 was provided}}
    A::g<3>(b);
    C::f<3>(b);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr241-C-f {{candidate template ignored: invalid explicitly-specified argument for template parameter 'T'}}
    C::g<3>(b);
    // expected-error@-1 {{no matching function for call to 'g'}}
    // expected-note@#dr241-C-g {{candidate template ignored: invalid explicitly-specified argument for template parameter 'T'}}
    using C::f;
    using C::g;
    f<3>(b);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr241-C-f {{candidate template ignored: invalid explicitly-specified argument for template parameter 'T'}}
    // expected-note@#dr241-A-f {{candidate function template not viable: requires 0 arguments, but 1 was provided}}
    g<3>(b);
  }
}

namespace dr243 { // dr243: yes
  struct B;
  struct A {
    A(B); // #dr243-A
  };
  struct B {
    operator A() = delete; // #dr243-B
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  } b;
  A a1(b);
  A a2 = b;
  // expected-error@-1 {{conversion from 'struct B' to 'A' is ambiguous}}
  // expected-note@#dr243-A {{candidate constructor}}
  // expected-note@#dr243-B {{candidate function has been explicitly deleted}}
}

namespace dr244 { // dr244: 11
                  // NB: this test is reused by dr399
  struct B {}; // #dr244-B
  struct D : B {};

  D D_object;
  typedef B B_alias;
  B* B_ptr = &D_object;

  void f() {
    D_object.~B();
    // expected-error@-1 {{destructor type 'dr244::B' in object destruction expression does not match the type 'D' of the object being destroyed}}
    // expected-note@#dr244-B {{type 'dr244::B' found by destructor name lookup}}
    D_object.B::~B();
    D_object.D::~B(); // FIXME: Missing diagnostic for this.
    B_ptr->~B();
    B_ptr->~B_alias();
    B_ptr->B_alias::~B();
    B_ptr->B_alias::~B_alias();
    B_ptr->dr244::~B();
    // expected-error@-1 {{qualified member access refers to a member in namespace 'dr244'}}
    B_ptr->dr244::~B_alias();
    // expected-error@-1 {{qualified member access refers to a member in namespace 'dr244'}}
  }

  template<typename T, typename U>
  void f(T *B_ptr, U D_object) {
    D_object.~B(); // FIXME: Missing diagnostic for this.
    D_object.B::~B();
    D_object.D::~B(); // FIXME: Missing diagnostic for this.
    B_ptr->~B();
    B_ptr->~B_alias();
    B_ptr->B_alias::~B();
    B_ptr->B_alias::~B_alias();
    B_ptr->dr244::~B();
    // expected-error@-1 {{'dr244' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T'}}
    B_ptr->dr244::~B_alias();
    // expected-error@-1 {{'dr244' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T'}}
  }
  template void f<B, D>(B*, D);

  namespace N {
    template<typename T> struct E {};
    typedef E<int> F;
  }
  void g(N::F f) {
    typedef N::F G; // #dr244-G
    f.~G();
    f.G::~E();
    // expected-error@-1 {{ISO C++ requires the name after '::~' to be found in the same scope as the name before '::~'}}
    f.G::~F();
    // expected-error@-1 {{undeclared identifier 'F' in destructor name}}
    f.G::~G();
    // This is technically ill-formed; E is looked up in 'N::' and names the
    // class template, not the injected-class-name of the class. But that's
    // probably a bug in the standard.
    f.N::F::~E();
    // expected-error@-1 {{ISO C++ requires the name after '::~' to be found in the same scope as the name before '::~'}}
    // This is valid; we look up the second F in the same scope in which we
    // found the first one, that is, 'N::'.
    f.N::F::~F();
    // This is technically ill-formed; G is looked up in 'N::' and is not found.
    // Rejecting this seems correct, but most compilers accept, so we do also.
    f.N::F::~G();
    // expected-error@-1 {{qualified destructor name only found in lexical scope; omit the qualifier to find this type name by unqualified lookup}}
    // expected-note@#dr244-G {{type 'G' (aka 'E<int>') found by destructor name lookup}}
  }

  // Bizarrely, compilers perform lookup in the scope for qualified destructor
  // names, if the nested-name-specifier is non-dependent. Ensure we diagnose
  // this.
  namespace QualifiedLookupInScope {
    namespace N {
      template <typename> struct S { struct Inner {}; };
    }
    template <typename U> void f(typename N::S<U>::Inner *p) {
      typedef typename N::S<U>::Inner T;
      p->::dr244::QualifiedLookupInScope::N::S<U>::Inner::~T();
      // expected-error@-1 {{no type named 'T' in 'dr244::QualifiedLookupInScope::N::S<int>'}}
      // expected-note@#dr244-f {{in instantiation of function template specialization 'dr244::QualifiedLookupInScope::f<int>' requested here}}
    }
    template void f<int>(N::S<int>::Inner *); // #dr244-f

    template <typename U> void g(U *p) {
      typedef U T;
      p->T::~T();
      p->U::~T();
      p->::dr244::QualifiedLookupInScope::N::S<int>::Inner::~T();
      // expected-error@-1 {{'T' does not refer to a type name in pseudo-destructor expression; expected the name of type 'U'}}
    }
    template void g(N::S<int>::Inner *);
  }
}

namespace dr245 { // dr245: yes
  struct S {
    enum E {}; // #dr245-E
    class E *p;
    // expected-error@-1 {{use of 'E' with tag type that does not match previous declaration}}
    // expected-note@#dr245-E {{previous use is here}}
  };
}

namespace dr246 { // dr246: 3.2
  struct S {
    S() try { // #dr246-try
      throw 0;
X: ;
    } catch (int) {
      goto X;
      // expected-error@-1 {{cannot jump from this goto statement to its label}}
      // expected-note@#dr246-try {{jump bypasses initialization of try block}}
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

namespace dr248 { // dr248: sup P1949
  int \u040d\u040e = 0;
}

namespace dr249 { // dr249: yes
  template<typename T> struct X { void f(); };
  template<typename T> void X<T>::f() {}
}

namespace dr250 { // dr250: yes
  typedef void (*FPtr)(double x[]);

  template<int I> void f(double x[]);
  FPtr fp = &f<3>;

  template<int I = 3> void g(double x[]);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  FPtr gp = &g<>;
}

namespace dr252 { // dr252: 3.1
  struct A {
    void operator delete(void*); // #dr252-A
  };
  struct B {
    void operator delete(void*); // #dr252-B
  };
  struct C : A, B {
    virtual ~C();
  };
  C::~C() {}
  // expected-error@-1 {{member 'operator delete' found in multiple base classes of different types}}
  // expected-note@#dr252-A {{member found by ambiguous name lookup}}
  // expected-note@#dr252-B {{member found by ambiguous name lookup}}

  struct D {
    void operator delete(void*, int); // #dr252-D
    virtual ~D();
  };
  D::~D() {}
  // expected-error@-1 {{no suitable member 'operator delete' in 'D'}}
  // expected-note@#dr252-D {{member 'operator delete' declared here}}

  struct E {
    void operator delete(void*, int);
    void operator delete(void*) = delete; // #dr252-E
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}} 
    virtual ~E();
  };
  E::~E() {}
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#dr252-E {{'operator delete' has been explicitly marked deleted here}}

  struct F {
    // If both functions are available, the first one is a placement delete.
    void operator delete(void*, size_t);
    void operator delete(void*) = delete; // #dr252-F
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    virtual ~F();
  };
  F::~F() {}
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#dr252-F {{'operator delete' has been explicitly marked deleted here}}

  struct G {
    void operator delete(void*, size_t);
    virtual ~G();
  };
  G::~G() {}
}

namespace dr254 { // dr254: 2.9
  template<typename T> struct A {
    typedef typename T::type type; // ok even if this is a typedef-name, because
                                   // it's not an elaborated-type-specifier
    typedef struct T::type foo;
    // expected-error@-1 {{typedef 'type' cannot be referenced with a struct specifier}}
    // expected-note@#dr254-instantiation {{in instantiation of template class 'dr254::A<dr254::C>' requested here}}
    // expected-note@#dr254-C {{declared here}}
  };
  struct B { struct type {}; };
  struct C { typedef struct {} type; }; // #dr254-C
  A<B>::type n;
  A<C>::type n; // #dr254-instantiation
}

namespace dr255 { // dr255: yes
struct S {
  void operator delete(void *){};
  void operator delete(void *, int){};
};
void f(S *p) { delete p; }
} // namespace dr255

// dr256: dup 624

namespace dr257 { // dr257: 3.4
  struct A { A(int); }; // #dr257-A
  struct B : virtual A {
    B() {}
    virtual void f() = 0;
  };
  struct C : B {
    C() {}
  };
  struct D : B {
    D() {}
    // expected-error@-1 {{constructor for 'dr257::D' must explicitly initialize the base class 'A' which does not have a default constructor}}
    // expected-note@#dr257-A {{'dr257::A' declared here}}
    void f();
  };
}

namespace dr258 { // dr258: 2.8
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
    template<int> int &g(int); // #dr258-B-g
    int &h();
  } b;
  int &w = b.f(0);
  int &x = b.g<int>(0);
  // expected-error@-1 {{no matching member function for call to 'g'}}
  // expected-note@#dr258-B-g {{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}
  int &y = b.h();
  float &z = const_cast<const B&>(b).h();

  struct C {
    virtual void f(const int) = 0;
  };
  struct D : C {
    void f(int);
  } d;

  struct E {
    virtual void f() = 0; // #dr258-E-f
  };
  struct F : E {
    void f() const {}
  } f;
  // expected-error@-1 {{variable type 'struct F' is an abstract class}}
  // expected-note@#dr258-E-f {{unimplemented pure virtual method 'f' in 'F'}}
}

namespace dr259 { // dr259: 4
  template<typename T> struct A {};
  template struct A<int>; // #dr259-A-int 
  template struct A<int>;
  // expected-error@-1 {{duplicate explicit instantiation of 'A<int>'}}
  // expected-note@#dr259-A-int {{previous explicit instantiation is here}}

  template<> struct A<float>; // #dr259-A-float
  template struct A<float>;
  // expected-warning@-1 {{explicit instantiation of 'A<float>' that occurs after an explicit specialization has no effect}}
  // expected-note@#dr259-A-float {{previous template specialization is here}}

  template struct A<char>; // #dr259-A-char
  template<> struct A<char>;
  // expected-error@-1 {{explicit specialization of 'dr259::A<char>' after instantiation}}
  // expected-note@#dr259-A-char {{explicit instantiation first required here}}

  template<> struct A<double>;
  template<> struct A<double>;
  template<> struct A<double> {}; // #dr259-A-double
  template<> struct A<double> {};
  // expected-error@-1 {{redefinition of 'A<double>'}}
  // expected-note@#dr259-A-double {{previous definition is here}}

  template<typename T> struct B; // #dr259-B
  template struct B<int>;
  // expected-error@-1 {{explicit instantiation of undefined template 'dr259::B<int>'}}
  // expected-note@#dr259-B {{template is declared here}}

  template<> struct B<float>; // #dr259-B-float
  template struct B<float>;
  // expected-warning@-1 {{explicit instantiation of 'B<float>' that occurs after an explicit specialization has no effect}}
  // expected-note@#dr259-B-float {{previous template specialization is here}}
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
    virtual ~C() {}
    // expected-warning@-1 {{'operator delete' was marked unused but was used}}
  };

  struct D {
    inline void operator delete(void*) __attribute__((unused));
  };
  void h() { C::operator delete(0); }
  // expected-warning@-1 {{'operator delete' was marked unused but was used}}

#pragma clang diagnostic pop
}

namespace dr262 { // dr262: yes
  int f(int = 0, ...);
  int k = f();
  int l = f(0);
  int m = f(0, 0);
}

namespace dr263 { // dr263: 3.3
  struct X {};
  struct Y {
#if __cplusplus < 201103L
    friend X::X() throw();
    friend X::~X() throw();
#elif __cplusplus <= 201703L
    friend constexpr X::X() noexcept;
    friend X::~X();
#else
    friend constexpr X::X() noexcept;
    friend constexpr X::~X();
#endif
    Y::Y();
    // expected-error@-1 {{extra qualification on member 'Y'}}
    Y::~Y();
    // expected-error@-1 {{extra qualification on member '~Y'}}
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
      ~X();
      // expected-error@-1 {{invalid argument type 'X' to unary expression}}
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
    template <class T> void f(T) {} // #dr275-N-f
    template <class T> void g(T) {} // #dr275-N-g
    template <> void f(int);
    template <> void f(char);
    template <> void f(double);
    template <> void g(char);
  }

  using namespace N;

  namespace M {
    template <> void N::f(char) {}
    // expected-error@-1 {{cannot define or redeclare 'f' here because namespace 'M' does not enclose namespace 'N'}}
    template <class T> void g(T) {}
    template <> void g(char) {}
    // FIXME: this should be rejected in c++98 too
    template void f(long);
    // since-cxx11-error@-1 {{explicit instantiation of 'dr275::N::f' must occur in namespace 'N'}}
    // since-cxx11-note@#dr275-N-f {{explicit instantiation refers here}}
    // FIXME: this should be rejected in c++98 too
    template void N::f(unsigned long);
    // since-cxx11-error@-1 {{explicit instantiation of 'f' not in a namespace enclosing 'N'}}
    // since-cxx11-note@#dr275-N-f {{explicit instantiation refers here}}
    template void h(long);
    // expected-error@-1 {{explicit instantiation of 'h' does not refer to a function template, variable template, member function, member class, or static data member}}
    template <> void f(double) {}
    // expected-error@-1 {{no function template matches function template specialization 'f'}}
  }

  template <class T> void g(T) {} // #dr275-g 

  template <> void N::f(char) {}
  template <> void f(int) {}
  // expected-error@-1 {{no function template matches function template specialization 'f'}}

  // FIXME: this should be rejected in c++98 too
  template void f(short);
  // since-cxx11-error@-1 {{explicit instantiation of 'dr275::N::f' must occur in namespace 'N'}}
  // since-cxx11-note@#dr275-N-f {{explicit instantiation refers here}}
  template void N::f(unsigned short);

  // FIXME: this should probably be valid. the wording from the issue
  // doesn't clarify this, but it follows from the usual rules.
  template void g(int);
  // expected-error@-1 {{partial ordering for explicit instantiation of 'g' is ambiguous}}
  // expected-note@#dr275-g {{explicit instantiation candidate function 'dr275::g<int>' template here [with T = int]}}
  // expected-note@#dr275-N-g {{explicit instantiation candidate function 'dr275::N::g<int>' template here [with T = int]}}

  // FIXME: likewise, this should also be valid.
  template<typename T> void f(T) {} // #dr275-f
  template void f(short);
  // expected-error@-1 {{partial ordering for explicit instantiation of 'f' is ambiguous}}
  // expected-note@#dr275-f {{explicit instantiation candidate function 'dr275::f<short>' template here [with T = short]}}
  // expected-note@#dr275-N-f {{explicit instantiation candidate function 'dr275::N::f<short>' template here [with T = short]}}
}

// dr276: na

namespace dr277 { // dr277: 3.1
  typedef int *intp;
  int *p = intp();
  int a[fold(intp() ? -1 : 1)];
}

namespace dr280 { // dr280: 2.9
  typedef void f0();
  typedef void f1(int);
  typedef void f2(int, int);
  typedef void f3(int, int, int);
  struct A {
    operator f1*(); // #dr280-A-f1
    operator f2*();
  };
  struct B {
    operator f0*(); // #dr280-B-f0
  private:
    operator f3*(); // #dr280-B-f3
  };
  struct C {
    operator f0*(); // #dr280-C-f0
    operator f1*(); // #dr280-C-f1
    operator f2*(); // #dr280-C-f2
    operator f3*(); // #dr280-C-f3
  };
  struct D : private A, B { // #dr280-D
    operator f2*(); // #dr280-D-f2
  } d;
  struct E : C, D {} e;
  void g() {
    d(); // ok, public
    d(0);
    // expected-error@-1 {{'operator void (*)(int)' is a private member of 'dr280::A'}}
    // expected-note@#dr280-D {{constrained by private inheritance here}}
    // expected-note@#dr280-A-f1 {{member is declared here}}
    d(0, 0); // ok, suppressed by member in D
    d(0, 0, 0);
    // expected-error@-1 {{'operator void (*)(int, int, int)' is a private member of 'dr280::B'}}
    // expected-note@#dr280-B-f3 {{declared private here}}
    e();
    // expected-error@-1 {{call to object of type 'struct E' is ambiguous}}
    // expected-note@#dr280-B-f0 {{conversion candidate of type 'void (*)()'}}
    // expected-note@#dr280-C-f0 {{conversion candidate of type 'void (*)()'}}
    e(0);
    // expected-error@-1 {{call to object of type 'struct E' is ambiguous}}
    // expected-note@#dr280-A-f1 {{conversion candidate of type 'void (*)(int)'}}
    // expected-note@#dr280-C-f1 {{conversion candidate of type 'void (*)(int)'}}
    e(0, 0);
    // expected-error@-1 {{call to object of type 'struct E' is ambiguous}}
    // expected-note@#dr280-C-f2 {{conversion candidate of type 'void (*)(int, int)'}}
    // expected-note@#dr280-D-f2 {{conversion candidate of type 'void (*)(int, int)'}}
    e(0, 0, 0);
    // expected-error@-1 {{call to object of type 'struct E' is ambiguous}}
    // expected-note@#dr280-B-f3 {{conversion candidate of type 'void (*)(int, int, int)'}}
    // expected-note@#dr280-C-f3 {{conversion candidate of type 'void (*)(int, int, int)'}}
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
  template<typename T> // #dr283-template
  struct S {
    friend class T;
    // expected-error@-1 {{declaration of 'T' shadows template parameter}}
    // expected-note@#dr283-template {{template parameter is declared here}}
    class T;
    // expected-error@-1 {{declaration of 'T' shadows template parameter}}
    // expected-note@#dr283-template {{template parameter is declared here}}
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
  struct B::V {};
  // expected-error@-1 {{no struct named 'V' in namespace 'dr284::B'}}
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
  struct D::V {};
  // expected-error@-1 {{no struct named 'V' in 'dr284::D'}}
  struct D::W {};
  struct D::X {}; // FIXME: ill-formed
  enum D::Y e2; // ok per dr417
  class D::Z z2; // ok per dr417
}

namespace dr285 { // dr285: yes
  template<typename T> void f(T, int); // #dr285-f-T-int
  template<typename T> void f(int, T); // #dr285-f-int-T
  template<> void f<int>(int, int) {}
  // expected-error@-1 {{function template specialization 'f' ambiguously refers to more than one function template; explicitly specify additional template arguments to identify a particular function template}}
  // expected-note@#dr285-f-int-T {{function template 'dr285::f<int>' matches specialization [with T = int]}}
  // expected-note@#dr285-f-T-int {{function template 'dr285::f<int>' matches specialization [with T = int]}}
}

namespace dr286 { // dr286: 2.8
  template<class T> struct A {
    class C {
      template<class T2> struct B {}; // #dr286-B 
    };
  };

  template<class T>
  template<class T2>
  struct A<T>::C::B<T2*> { };

  A<short>::C::B<int*> absip;
  // expected-error@-1 {{'B' is a private member of 'dr286::A<short>::C'}}
  // expected-note@#dr286-B {{implicitly declared private here}}
}

// dr288: na

namespace dr289 { // dr289: yes
  struct A; // #dr289-A
  struct B : A {};
  // expected-error@-1 {{base class has incomplete type}}
  // expected-note@#dr289-A {{forward declaration of 'dr289::A'}}

  template<typename T> struct C { typename T::error error; };
  // expected-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
  // expected-note@#dr289-C-int {{in instantiation of template class 'dr289::C<int>' requested here}}
  struct D : C<int> {}; // #dr289-C-int
}

// dr290: na
// dr291: dup 391
// dr292 FIXME: write a codegen test

namespace dr294 { // dr294: no
  void f() throw(int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  int main() {
    (void)static_cast<void (*)() throw()>(f); // FIXME: ill-formed in C++14 and before
    // FIXME: since-cxx17-error@-1 {{static_cast from 'void (*)() throw(int)' to 'void (*)() throw()' is not allowed}}
    //
    // Irony: the above is valid in C++17 and beyond, but that's exactly when
    // we reject it. In C++14 and before, this is ill-formed because an
    // exception-specification is not permitted in a type-id. In C++17, this is
    // valid because it's the inverse of a standard conversion sequence
    // containing a function pointer conversion. (Well, it's actually not valid
    // yet, as a static_cast is not permitted to reverse a function pointer
    // conversion, but that is being changed by core issue).
    (void)static_cast<void (*)() throw(int)>(f); // FIXME: ill-formed in C++14 and before
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}

    void (*p)() throw() = f;
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{cannot initialize a variable of type 'void (*)() throw()' with an lvalue of type 'void () throw(int)': different exception specifications}}
    void (*q)() throw(int) = f;
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  }
}

namespace dr295 { // dr295: 3.7
  typedef int f();
  const f g;
  // expected-warning@-1 {{'const' qualifier on function type 'f' (aka 'int ()') has no effect}}
  f &r = g;
  template<typename T> struct X {
    const T &f;
  };
  X<f> x = {g};

  typedef int U();
  typedef const U U;
  // expected-warning@-1 {{'const' qualifier on function type 'U' (aka 'int ()') has no effect}}

  typedef int (*V)();
  typedef volatile U *V;
  // expected-warning@-1 {{'volatile' qualifier on function type 'U' (aka 'int ()') has no effect}}
}

namespace dr296 { // dr296: yes
  struct A {
    static operator int() { return 0; }
    // expected-error@-1 {{conversion function must be a non-static member function}}
  };
}

namespace dr298 { // dr298: 3.1
  struct A {
    typedef int type;
    A();
    ~A();
  };
  typedef A B; // #dr298-B
  typedef const A C; // #dr298-C

  A::type i1;
  B::type i2;
  C::type i3;

  struct A a;
  struct B b;
  // expected-error@-1 {{typedef 'B' cannot be referenced with a struct specifier}}
  // expected-note@#dr298-B {{declared here}}
  struct C c;
  // expected-error@-1 {{typedef 'C' cannot be referenced with a struct specifier}}
  // expected-note@#dr298-C {{declared here}}

  B::B() {}
  // expected-error@-1 {{a type specifier is required for all declarations}}
  B::A() {} // ok
  C::~C() {}
  // expected-error@-1 {{destructor cannot be declared using a typedef 'C' (aka 'const dr298::A') of the class name}}

  typedef struct D E; // #dr298-E
  struct E {};
  // expected-error@-1 {{definition of type 'E' conflicts with typedef of the same name}}
  // expected-note@#dr298-E {{'E' declared here}}

  struct F {
    ~F();
  };
  typedef const F G;
  G::~F() {} // ok
}

namespace dr299 { // dr299: 2.8 c++11
  struct S {
    operator int();
  };
  struct T {
    operator int(); // #dr299-int
    operator unsigned short(); // #dr299-ushort
  };
  // FIXME: should this apply to c++98 mode?
  int *p = new int[S()];
  // cxx98-error@-1 {{implicit conversion from array size expression of type 'S' to integral type 'int' is a C++11 extension}}
  int *q = new int[T()]; // #dr299-q
  // cxx98-11-error@#dr299-q {{ambiguous conversion of array size expression of type 'T' to an integral or enumeration type}}
  //  cxx98-11-note@#dr299-int {{conversion to integral type 'int' declared here}}
  //  cxx98-11-note@#dr299-ushort {{conversion to integral type 'unsigned short' declared here}}
  // since-cxx14-error-re@#dr299-q {{{{conversion from 'T' to 'unsigned (long long|long|int)' is ambiguous}}}}
  //  since-cxx14-note@#dr299-int {{candidate function}}
  //  since-cxx14-note@#dr299-ushort {{candidate function}}
}
