// RUN: %clang_cc1 -fsyntax-only -verify %s 
struct X {
  operator bool();
};

int& f(bool);
float& f(int);

void f_test(X x) {
  int& i1 = f(x);
}

struct Y {
  operator short();
  operator float();
};

void g(int);

void g_test(Y y) {
  g(y);
  short s;
  s = y;
}

struct A { };
struct B : A { };

struct C {
  operator B&();
};

// Test reference binding via an lvalue conversion function.
void h(volatile A&);
void h_test(C c) {
  h(c);
}

// Test conversion followed by copy-construction
struct FunkyDerived;

struct Base { 
  Base(const FunkyDerived&);
};

struct Derived : Base { };

struct FunkyDerived : Base { };

struct ConvertibleToBase {
  operator Base();
};

struct ConvertibleToDerived {
  operator Derived();
};

struct ConvertibleToFunkyDerived {
  operator FunkyDerived();
};

void test_conversion(ConvertibleToBase ctb, ConvertibleToDerived ctd,
                     ConvertibleToFunkyDerived ctfd) {
  Base b1 = ctb;
  Base b2(ctb);
  Base b3 = ctd;
  Base b4(ctd);
  Base b5 = ctfd;
}

struct X1 {
  X1(X1&); // expected-note{{candidate constructor not viable: expects an lvalue for 1st argument}}
};

struct X2 {
  operator X1();
};

int &f(X1);
float &f(...);

void g(X2 b) {
  int &ir = f(b); // expected-error{{no viable constructor copying parameter of type 'X1'}}
}

namespace rdar10202900 {
  class A {
  public:
    A();

  private:
    A(int i); // expected-note{{declared private here}}
  };

  void testA(A a) {
    int b = 10;
    a = b; // expected-error{{calling a private constructor of class 'rdar10202900::A'}}
  }
}
