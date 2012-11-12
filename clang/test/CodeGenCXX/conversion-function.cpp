// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s
// XFAIL: *
extern "C" int printf(...);
struct S {
  operator int();
};

S::operator int() {
  return 10;
}

int f(S s) {
  return s;
}

class X { // ...
  public: operator int() { printf("operator int()\n"); return iX; }
  public: operator float() { printf("operator float()\n"); return fX; }
  X() : iX(100), fX(1.234)  {}
  int iX;
  float fX;
};

X x;

struct Z {
    operator X() { printf("perator X()\n"); x.iX += iZ; x.fX += fZ; return x; }
    int iZ;
    float fZ;
    Z() : iZ(1), fZ(1.00) {}
};

Z z;

class Y { // ...
  public: operator Z(){printf("perator Z()\n"); return z; }
};

Y y;

int count=0;
class O { // ...
public: 
  operator int(){ return ++iO; }
  O() : iO(count++) {}
  int iO;
};

void g(O a, O b) {
  int i = (a) ? 1+a : 0; 
  int j = (a&&b) ? a+b : i; 
  if (a) { }
  printf("i = %d j = %d a.iO = %d b.iO = %d\n", i, j, a.iO, b.iO);
}

int main() {
  int c = X(Z(y)); // OK: y.operator Z().operator X().operator int()
  printf("c = %d\n", c);
  float f = X(Z(y));
  printf("f = %f\n", f);
  int i = x;
  printf("i = %d float = %f\n", i, float(x));
  i = int(X(Z(y)));
  f = float(X(Z(y)));
  printf("i = %d float = %f\n", i,f);
  f = (float)x;
  i = (int)x;
  printf("i = %d float = %f\n", i,f);

  int d = (X)((Z)y);
  printf("d = %d\n", d);

  int e = (int)((X)((Z)y));
  printf("e = %d\n", e);
  O o1, o2;
  g(o1, o2);
}

// Test. Conversion in base class is visible in derived class.
class XB {
  int a;
public:
  operator int();
};

class Yb : public XB {
  double b;
public:
  operator char();
};

void f(Yb& a) {
  int i = a; // OK. calls XB::operator int();
  char ch = a;  // OK. calls Yb::operator char();
}

struct A {
  operator int() const;
};

// CHECK-LP64: .globl __ZN1ScviEv
// CHECK-LP64-NEXT: __ZN1ScviEv:
// CHECK-LP64: callq __ZN1Ycv1ZEv
// CHECK-LP64: callq __ZN1Zcv1XEv
// CHECK-LP64: callq __ZN1XcviEv
// CHECK-LP64: callq __ZN1XcvfEv
// CHECK-LP64: callq __ZN2XBcviEv
// CHECK-LP64: callq __ZN2YbcvcEv

// CHECK-LP32: .globl  __ZN1ScviEv
// CHECK-LP32-NEXT: __ZN1ScviEv:
// CHECK-LP32: call L__ZN1Ycv1ZEv
// CHECK-LP32: call L__ZN1Zcv1XEv
// CHECK-LP32: call L__ZN1XcviEv
// CHECK-LP32: call L__ZN1XcvfEv
// CHECK-LP32: call L__ZN2XBcviEv
// CHECK-LP32: call L__ZN2YbcvcEv
