// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar: // 8379892

struct X {
  X();
  X(const X&);
  ~X();

  static int staticData;
  int data;
  void method();
};

@interface A {
  X xval;
}

- (X)x;
- (void)setx:(X)x;
@end

void f(A* a) {
  a.x = X(); // expected-error {{no setter method 'setX:' for assignment to property}}
}

struct Y : X { };

@interface B {
@private
  Y *y;
}
- (Y)value;
- (void)setValue : (Y) arg;
@property Y value;
@end

void g(B *b) {
  b.value.data = 17; // expected-error {{not assignable}}
  b.value.staticData = 17;
  b.value.method();
}

@interface C
@end

@implementation C
- (void)method:(B *)b {
  // <rdar://problem/8985943>
  b.operator+ = 17; // expected-error{{'operator+' is not a valid property name (accessing an object of type 'B *')}}
  b->operator+ = 17; // expected-error{{'B' does not have a member named 'operator+'}}
}
@end

// PR9759
class Forward;
@interface D { // expected-note 2 {{'D' declared here}}
@public
  int ivar;
}

@property int property;
@end

void testD(D *d) {
  d.Forward::property = 17; // expected-error{{property access cannot be qualified with 'Forward::'}}
  d->Forward::ivar = 12; // expected-error{{instance variable access cannot be qualified with 'Forward::'}}
  d.D::property = 17; // expected-error{{'D' is not a class, namespace, or enumeration}}
  d->D::ivar = 12; // expected-error{{'D' is not a class, namespace, or enumeration}}
}
