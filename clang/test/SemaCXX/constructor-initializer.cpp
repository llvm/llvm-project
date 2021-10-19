// RUN: %clang_cc1 -Wreorder -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wreorder -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -Wreorder -fsyntax-only -verify -std=c++11 %s

class A { 
  int m;
public:
   A() : A::m(17) { } // expected-error {{member initializer 'm' does not name a non-static data member or base class}}
   A(int);
};

class B : public A { 
public:
  B() : A(), m(1), n(3.14) { }

private:
  int m;
  float n;  
};


class C : public virtual B { 
public:
  C() : B() { }
};

class D : public C { 
public:
  D() : B(), C() { }
};

class E : public D, public B {  // expected-warning{{direct base 'B' is inaccessible due to ambiguity:\n    class E -> class D -> class C -> class B\n    class E -> class B}}
public:
  E() : B(), D() { } // expected-error{{base class initializer 'B' names both a direct base class and an inherited virtual base class}}
};


typedef int INT;

class F : public B { 
public:
  int B;

  F() : B(17),
        m(17), // expected-error{{member initializer 'm' does not name a non-static data member or base class}}
        INT(17) // expected-error{{constructor initializer 'INT' (aka 'int') does not name a class}}
  { 
  }
};

class G : A {
  G() : A(10); // expected-error{{expected '{'}}
};

void f() : a(242) { } // expected-error{{only constructors take base initializers}}

class H : A {
  H();
};

H::H() : A(10) { }


class  X {};
class Y {};

struct S : Y, virtual X {
  S (); 
};

struct Z : S { 
  Z() : X(), S(), E()  {} // expected-error {{type 'E' is not a direct or virtual base of 'Z'}}
};

class U { 
  union { int a; char* p; };
  union { int b; double d; };

  U() :  a(1), // expected-note {{previous initialization is here}}
         p(0), // expected-error {{initializing multiple members of union}}
         d(1.0)  {}
};

struct V {};
struct Base {};
struct Base1 {};

struct Derived : Base, Base1, virtual V {
  Derived ();
};

struct Current : Derived {
  int Derived;
  Current() : Derived(1), ::Derived(), // expected-warning {{initializer order does not match the declaration order}} \
                                       // expected-note {{field 'Derived' will be initialized after base '::Derived'}} \
                                       // expected-note {{base class '::Derived' will be initialized after base 'Derived::V'}}
              ::Derived::Base(),       // expected-error {{type '::Derived::Base' is not a direct or virtual base of 'Current'}}
              Derived::Base1(),        // expected-error {{type 'Derived::Base1' is not a direct or virtual base of 'Current'}}
              Derived::V(),
              ::NonExisting(),      // expected-error {{member initializer 'NonExisting' does not name a non-static data member or}}
              INT::NonExisting() {} // expected-error {{'INT' (aka 'int') is not a class, namespace, or enumeration}} \
                                                  // expected-error {{member initializer 'NonExisting' does not name a non-static data member or}}
};

struct M {              // expected-note 2 {{candidate constructor (the implicit copy constructor)}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 2 {{candidate constructor (the implicit move constructor) not viable}}
#endif
// expected-note@-4 2 {{'M' declared here}}
  M(int i, int j);      // expected-note 2 {{candidate constructor}}
};

struct N : M  {
  N() : M(1),        // expected-error {{no matching constructor for initialization of 'M'}}
        m1(100) {  } // expected-error {{no matching constructor for initialization of 'M'}}
  M m1;
};

struct P : M  {
  P()  {  } // expected-error {{constructor for 'P' must explicitly initialize the base class 'M' which does not have a default constructor}} \
            // expected-error {{member 'm'}}
  M m; // expected-note {{member is declared here}}
};

struct Q {
  Q() : f1(1,2),       // expected-error {{excess elements in scalar initializer}}
        pf(0.0)  { }   // expected-error {{cannot initialize a member subobject of type 'float *' with an rvalue of type 'double'}}
  float f1;

  float *pf;
};

// A silly class used to demonstrate field-is-uninitialized in constructors with
// multiple params.
int IntParam(int i) { return 0; };
class TwoInOne { public: TwoInOne(TwoInOne a, TwoInOne b) {} };
class InitializeUsingSelfTest {
  bool A;
  char* B;
  int C;
  TwoInOne D;
  int E;
  InitializeUsingSelfTest(int F)
      : A(A),  // expected-warning {{field 'A' is uninitialized when used here}}
        B((((B)))),  // expected-warning {{field 'B' is uninitialized when used here}}
        C(A && InitializeUsingSelfTest::C),  // expected-warning {{field 'C' is uninitialized when used here}}
        D(D,  // expected-warning {{field 'D' is uninitialized when used here}}
          D), // expected-warning {{field 'D' is uninitialized when used here}}
        E(IntParam(E)) {} // expected-warning {{field 'E' is uninitialized when used here}}
};

int IntWrapper(int &i) { return 0; };
class InitializeUsingSelfExceptions {
  int A;
  int B;
  int C;
  void *P;
  InitializeUsingSelfExceptions(int B)
      : A(IntWrapper(A)),  // Due to a conservative implementation, we do not report warnings inside function/ctor calls even though it is possible to do so.
        B(B),  // Not a warning; B is a local variable.
        C(sizeof(C)),  // sizeof doesn't reference contents, do not warn
        P(&P) {} // address-of doesn't reference contents (the pointer may be dereferenced in the same expression but it would be rare; and weird)
};

class CopyConstructorTest {
  bool A, B, C;
  CopyConstructorTest(const CopyConstructorTest& rhs)
      : A(rhs.A),
        B(B),  // expected-warning {{field 'B' is uninitialized when used here}}
        C(rhs.C || C) { }  // expected-warning {{field 'C' is uninitialized when used here}}
};

// Make sure we aren't marking default constructors when we shouldn't be.
template<typename T>
struct NDC {
  T &ref;
  
  NDC() { }
  NDC(T &ref) : ref(ref) { }
};
  
struct X0 : NDC<int> {
  X0(int &ref) : NDC<int>(ref), ndc(ref) { }
  
  NDC<int> ndc;
};

namespace Test0 {

struct A { A(); };

struct B {
  B() { } 
  const A a;
};

}

namespace Test1 {
  struct A {
    enum Kind { Foo } Kind;
    A() : Kind(Foo) {}
  };
}

namespace Test2 {

struct A { 
  A(const A&);
};

struct B : virtual A { };

  struct C : A, B { }; // expected-warning{{direct base 'Test2::A' is inaccessible due to ambiguity:\n    struct Test2::C -> struct Test2::A\n    struct Test2::C -> struct Test2::B -> struct Test2::A}}

C f(C c) {
  return c;
}

}

// Don't build implicit initializers for anonymous union fields when we already
// have an explicit initializer for another field in the union.
namespace PR7402 {
  struct S {
    union {
      void* ptr_;
      struct { int i_; };
    };

    template <typename T> S(T) : ptr_(0) { }
  };

  void f() {
    S s(3);
  }
}

// <rdar://problem/8308215>: don't crash.
// Lots of questionable recovery here;  errors can change.
namespace test3 {
  class A : public std::exception {}; // expected-error {{undeclared identifier}} expected-error {{expected class name}}
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
  // expected-note@-3 {{candidate constructor (the implicit move constructor) not viable}}
#endif
  // expected-note@-5 {{candidate constructor (the implicit default constructor) not viable}}

  class B : public A {
  public:
    B(const String& s, int e=0) // expected-error {{unknown type name}} 
      : A(e), m_String(s) , m_ErrorStr(__null) {} // expected-error {{no matching constructor}} expected-error {{does not name}}
    B(const B& e)
      : A(e), m_String(e.m_String), m_ErrorStr(__null) { // expected-error 2{{does not name}} \
      // expected-error {{no member named 'm_String' in 'test3::B'}}
    }
  };
}

// PR8075
namespace PR8075 {

struct S1 {
  enum { FOO = 42 };
  static const int bar = 42;
  static int baz();
  S1(int);
};

const int S1::bar;

struct S2 {
  S1 s1;
  S2() : s1(s1.FOO) {}
};

struct S3 {
  S1 s1;
  S3() : s1(s1.bar) {}
};

struct S4 {
  S1 s1;
  S4() : s1(s1.baz()) {}
};

}

namespace PR12049 {
  int function();

  class Class
  {
  public:
      Class() : member(function() {} // expected-note {{to match this '('}}

      int member; // expected-error {{expected ')'}}
  };
}

namespace PR14073 {
  struct S1 { union { int n; }; S1() : n(n) {} };  // expected-warning {{field 'n' is uninitialized when used here}}
  struct S2 { union { union { int n; }; char c; }; S2() : n(n) {} };  // expected-warning {{field 'n' is uninitialized when used here}}
  struct S3 { struct { int n; }; S3() : n(n) {} };  // expected-warning {{field 'n' is uninitialized when used here}}
}

namespace PR10758 {
struct A;
struct B {
  B (A const &); // expected-note 2 {{candidate constructor not viable: no known conversion from 'const PR10758::B' to 'const PR10758::A &' for 1st argument}}
  B (B &); // expected-note 2 {{candidate constructor not viable: 1st argument ('const PR10758::B') would lose const qualifier}}
};
struct A {
  A (B); // expected-note 2 {{passing argument to parameter here}}
};

B f(B const &b) {
  return b; // expected-error {{no matching constructor for initialization of 'PR10758::B'}}
}

A f2(const B &b) {
  return b; // expected-error {{no matching constructor for initialization of 'PR10758::B'}}
}
}
