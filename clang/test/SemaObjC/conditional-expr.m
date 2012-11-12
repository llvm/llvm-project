// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -Wno-objc-root-class %s
@protocol NSObject
@end

@protocol DTOutputStreams <NSObject>
@end

@interface DTFilterOutputStream <DTOutputStreams>
- nextOutputStream;
@end

@implementation DTFilterOutputStream
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream];
  self = nextOutputStream;
  return nextOutputStream ? nextOutputStream : self;
}
- nextOutputStream {
  return self;
}
@end

@interface DTFilterOutputStream2
- nextOutputStream; // expected-note {{method definition for 'nextOutputStream' not found}}
@end

@implementation DTFilterOutputStream2 // expected-warning {{incomplete implementation}}
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream];
  self = nextOutputStream; // expected-warning {{assigning to 'DTFilterOutputStream2 *' from incompatible type 'id<DTOutputStreams>'}}
  return nextOutputStream ? nextOutputStream : self; // expected-warning {{incompatible operand types ('id<DTOutputStreams>' and 'DTFilterOutputStream2 *')}}
}
@end

// No @interface declaration for DTFilterOutputStream3
@implementation DTFilterOutputStream3 // expected-warning {{cannot find interface declaration for 'DTFilterOutputStream3'}}
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream]; // expected-warning {{method '-nextOutputStream' not found (return type defaults to 'id')}}
  self = nextOutputStream; // expected-warning {{assigning to 'DTFilterOutputStream3 *' from incompatible type 'id<DTOutputStreams>'}}
  return nextOutputStream ? nextOutputStream : self; // expected-warning {{incompatible operand types ('id<DTOutputStreams>' and 'DTFilterOutputStream3 *')}}
}
@end

//

@protocol P0
@property int intProp;
@end
@protocol P1
@end
@protocol P2
@end

@interface A <P0>
@end

@interface B : A
@end

@interface C
@end

@interface D
@end

void f0(id<P0> x) {
  x.intProp = 1;
}

void f1(int cond, id<P0> x, id<P0> y) {
  (cond ? x : y).intProp = 1;
}

void f2(int cond, id<P0> x, A *y) {
  (cond ? x : y).intProp = 1;
}

void f3(int cond, id<P0> x, B *y) {
  (cond ? x : y).intProp = 1;
}

void f4(int cond, id x, B *y) {
  (cond ? x : y).intProp = 1; // expected-error {{property 'intProp' not found on object of type 'id'}}
}

void f5(int cond, id<P0> x, C *y) {
  (cond ? x : y).intProp = 1; // expected-warning {{incompatible operand types ('id<P0>' and 'C *')}} expected-error {{property 'intProp' not found on object of type 'id'}}
}

void f6(int cond, C *x, D *y) {
  (cond ? x : y).intProp = 1; // expected-warning {{incompatible operand types}}, expected-error {{property 'intProp' not found on object of type 'id'}}
}

id f7(int a, id<P0> x, A* p) {
  return a ? x : p;
}

void f8(int a, A<P0> *x, A *y) {
  [ (a ? x : y ) intProp ];
}

void f9(int a, A<P0> *x, A<P1> *y) {
  id l0 = (a ? x : y ); // expected-warning {{incompatible operand types ('A<P0> *' and 'A<P1> *')}}
  A<P0> *l1 = (a ? x : y ); // expected-warning {{incompatible operand types ('A<P0> *' and 'A<P1> *')}}
  A<P1> *l2 = (a ? x : y ); // expected-warning {{incompatible operand types ('A<P0> *' and 'A<P1> *')}}
  [ (a ? x : y ) intProp ]; // expected-warning {{incompatible operand types ('A<P0> *' and 'A<P1> *')}}
}

void f10(int a, id<P0> x, id y) {
  [ (a ? x : y ) intProp ];
}

void f11(int a, id<P0> x, id<P1> y) {
  [ (a ? x : y ) intProp ]; // expected-warning {{incompatible operand types ('id<P0>' and 'id<P1>')}}
}

void f12(int a, A<P0> *x, A<P1> *y) {
  A<P1>* l0 = (a ? x : y ); // expected-warning {{incompatible operand types ('A<P0> *' and 'A<P1> *')}}
}
