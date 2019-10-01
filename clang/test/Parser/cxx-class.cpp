// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -fcxx-exceptions %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -fcxx-exceptions -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -fcxx-exceptions -std=c++11 %s

class C;
class C {
public:
protected:
  typedef int A,B;
  static int sf(), u;

  struct S {};
  enum {}; // expected-warning{{declaration does not declare anything}}
  int; // expected-warning {{declaration does not declare anything}}
  int : 1, : 2;

public:
  void m0() {}; // ok, one extra ';' is permitted
  void m1() {}
  ; // ok, one extra ';' is permitted
  void m() {
    int l = 2;
  };; // expected-warning{{extra ';' after member function definition}}

  template<typename T> void mt(T) { }
  ;
  ; // expected-warning{{extra ';' inside a class}}

  virtual int vf() const volatile = 0;

  virtual int vf0() = 0l; // expected-error {{does not look like a pure-specifier}}
  virtual int vf1() = 1; // expected-error {{does not look like a pure-specifier}}
  virtual int vf2() = 00; // expected-error {{does not look like a pure-specifier}}
  virtual int vf3() = 0x0; // expected-error {{does not look like a pure-specifier}}
  virtual int vf4() = 0.0; // expected-error {{does not look like a pure-specifier}}
  virtual int vf5(){0}; // expected-error +{{}} expected-warning {{unused}}
  virtual int vf5a(){0;}; // function definition, expected-warning {{unused}}
  virtual int vf6()(0); // expected-error +{{}} expected-note +{{}}
  virtual int vf7() = { 0 }; // expected-error {{does not look like a pure-specifier}}
  
private:
  int x,f(),y,g();
  inline int h();
  static const int sci = 10;
  mutable int mi;
};
void glo()
{
  struct local {};
}

// PR3177
typedef union {
  __extension__ union {
    int a;
    float b;
  } y;
} bug3177;

// check that we don't consume the token after the access specifier 
// when it's not a colon
class D {
public // expected-error{{expected ':'}}
  int i;
};

// consume the token after the access specifier if it's a semicolon 
// that was meant to be a colon
class E {
public; // expected-error{{expected ':'}}
  int i;
};

class F {
    int F1 { return 1; }
#if __cplusplus <= 199711L
    // expected-error@-2 {{function definition does not declare parameters}}
#else
    // expected-error@-4 {{expected expression}}
    // expected-error@-5 {{expected}}
    // expected-note@-6 {{to match this '{'}}
    // expected-error@-7 {{expected ';' after class}}
#endif

    void F2 {}
#if __cplusplus <= 199711L
    // expected-error@-2 {{function definition does not declare parameters}}
#else
    // expected-error@-4 {{variable has incomplete type 'void'}}
    // expected-error@-5 {{expected ';' after top level declarator}}
#endif

    typedef int F3() { return 0; } // expected-error{{function definition declared 'typedef'}}
    typedef void F4() {} // expected-error{{function definition declared 'typedef'}}
};
#if __cplusplus >= 201103L
// expected-error@-2 {{extraneous closing brace}}
#endif

namespace ctor_error {
  class Foo {};
  // By [class.qual]p2, this is a constructor declaration.
  Foo::Foo (F) = F(); // expected-error{{does not match any declaration in 'ctor_error::Foo'}}

  class Ctor { // expected-note{{not complete until the closing '}'}}
    Ctor(f)(int); // ok
    Ctor(g(int)); // ok
    Ctor(x[5]); // expected-error{{incomplete type}}

    Ctor(UnknownType *); // expected-error{{unknown type name 'UnknownType'}}
    void operator+(UnknownType*); // expected-error{{unknown type name 'UnknownType'}}
  };

  Ctor::Ctor (x) = { 0 }; // \
    // expected-error{{qualified reference to 'Ctor' is a constructor name}}

  Ctor::Ctor(UnknownType *) {} // \
    // expected-error{{unknown type name 'UnknownType'}}
  void Ctor::operator+(UnknownType*) {} // \
    // expected-error{{unknown type name 'UnknownType'}}
}

namespace nns_decl {
  struct A {
    struct B;
  };
  namespace N {
    union C;
  }
  struct A::B; // expected-error {{forward declaration of struct cannot have a nested name specifier}}
  union N::C; // expected-error {{forward declaration of union cannot have a nested name specifier}}
}

// PR13775: Don't assert here.
namespace PR13775 {
  class bar
  {
   public:
    void foo ();
    void baz ();
  };
  void bar::foo ()
  {
    baz x(); // expected-error 3{{}}
  }
}

class pr16989 {
  void tpl_mem(int *) {
    return;
    class C2 {
      void f();
    };
    void C2::f() {} // expected-error{{function definition is not allowed here}}
  };
};

namespace CtorErrors {
  struct A {
    A(NonExistent); // expected-error {{unknown type name 'NonExistent'}}
  };
  struct B {
    B(NonExistent) : n(0) {} // expected-error {{unknown type name 'NonExistent'}}
    int n;
  };
  struct C {
    C(NonExistent) try {} catch (...) {} // expected-error {{unknown type name 'NonExistent'}}
  };
  struct D {
    D(NonExistent) {} // expected-error {{unknown type name 'NonExistent'}}
  };
}

namespace DtorErrors {
  struct A { ~A(); int n; } a;
  ~A::A() { n = 0; } // expected-error {{'~' in destructor name should be after nested name specifier}} expected-note {{previous}}
  A::~A() {} // expected-error {{redefinition}}

  struct B { ~B(); } *b;
  DtorErrors::~B::B() {} // expected-error {{'~' in destructor name should be after nested name specifier}}

  void f() {
    a.~A::A(); // expected-error {{'~' in destructor name should be after nested name specifier}}
    b->~DtorErrors::~B::B(); // expected-error {{'~' in destructor name should be after nested name specifier}}
  }

  struct C; // expected-note {{forward decl}}
  ~C::C() {} // expected-error {{incomplete}} expected-error {{'~' in destructor name should be after nested name specifier}}

  struct D { struct X {}; ~D() throw(X); };
  ~D::D() throw(X) {} // expected-error {{'~' in destructor name should be after nested name specifier}}

  ~Undeclared::Undeclared() {} // expected-error {{use of undeclared identifier 'Undeclared'}} expected-error {{'~' in destructor name should be after nested name specifier}}
  ~Undeclared:: {} // expected-error {{expected identifier}} expected-error {{'~' in destructor name should be after nested name specifier}}

  struct S {
    // For another struct's destructor, emit the same diagnostic like for
    // A::~A() in addition to the "~ in the wrong place" one.
    ~A::A() {} // expected-error {{'~' in destructor name should be after nested name specifier}} expected-error {{non-friend class member '~A' cannot have a qualified name}}
    A::~A() {} // expected-error {{non-friend class member '~A' cannot have a qualified name}}

    // An inline destructor with a redundant class name should also get the
    // same diagnostic as S::~S.
    ~S::S() {} // expected-error {{'~' in destructor name should be after nested name specifier}} expected-error {{extra qualification on member '~S'}}

    // This just shouldn't crash.
    int I; // expected-note {{declared here}}
    ~I::I() {} // expected-error {{'I' is not a class, namespace, or enumeration}} expected-error {{'~' in destructor name should be after nested name specifier}}
  };

  struct T {};
  T t1 = t1.T::~T<int>; // expected-error {{destructor name 'T' does not refer to a template}} expected-error {{expected '(' for function-style cast or type construction}} expected-error {{expected expression}}
  // Emit the same diagnostic as for the previous case, plus something about ~.
  T t2 = t2.~T::T<int>; // expected-error {{'~' in destructor name should be after nested name specifier}} expected-error {{destructor name 'T' does not refer to a template}} expected-error {{expected '(' for function-style cast or type construction}} expected-error {{expected expression}}
}

namespace BadFriend {
  struct A {
    friend int : 3; // expected-error {{friends can only be classes or functions}}
    friend void f() = 123; // expected-error {{illegal initializer}}
    friend virtual void f(); // expected-error {{'virtual' is invalid in friend declarations}}
    friend void f() final; // expected-error {{'final' is invalid in friend declarations}}
    friend void f() override; // expected-error {{'override' is invalid in friend declarations}}
  };
}

class PR20760_a {
  int a = ); // expected-error {{expected expression}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

  int b = }; // expected-error {{expected expression}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

  int c = ]; // expected-error {{expected expression}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

};
class PR20760_b {
  int d = d); // expected-error {{expected ';'}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

  int e = d]; // expected-error {{expected ';'}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

  int f = d // expected-error {{expected ';'}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{in-class initialization of non-static data member is a C++11 extension}}
#endif

};

namespace PR20887 {
class X1 { a::operator=; }; // expected-error {{undeclared identifier 'a'}}
class X2 { a::a; }; // expected-error {{undeclared identifier 'a'}}
}

class BadExceptionSpec {
  void f() throw(int; // expected-error {{expected ')'}} expected-note {{to match}}
  void g() throw(
      int(
          ; // expected-error {{unexpected ';' before ')'}}
          ));
};

namespace PR41192 {
extern struct A a;
struct A {} ::PR41192::a; // ok, no missing ';' here  expected-warning {{extra qualification}}

#if __cplusplus >= 201103L
struct C;
struct D { static C c; };
struct C {} decltype(D())::c; // expected-error {{'decltype' cannot be used to name a declaration}}
#endif
}

namespace ArrayMemberAccess {
  struct A {
    int x;
    template<typename T> int f() const;
  };
  void f(const A (&a)[]) {
    // OK: not a template-id.
    bool cond = a->x < 10 && a->x > 0;
    // OK: a template-id.
    a->f<int>();
  }
}

// PR11109 must appear at the end of the source file
class pr11109r3 { // expected-note{{to match this '{'}}
  public // expected-error{{expected ':'}} expected-error{{expected '}'}} expected-error{{expected ';' after class}}
