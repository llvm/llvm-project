// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P0
@end
@protocol P1
@end
@protocol P2
@end

@interface A <P0>
@end

@interface B : A
@end

void bar(id x);
void barP0(id<P0> x);
void barP1(id<P1> x);
void barP2(id<P2> x);

void f0(A *a) {
  id l = a;
}

void f1(id x, A *a) {
  id<P0> l = a;
}

void f2(id<P1> x) {
  id<P0> l = x; // expected-error {{cannot initialize a variable of type 'id<P0>' with an lvalue of type 'id<P1>'}}
}

void f3(A *a) {
  id<P1> l = a; // expected-error {{cannot initialize a variable of type 'id<P1>' with an lvalue of type 'A *'}}
}

void f4(int cond, id x, A *a) {
  bar(cond ? x : a);
}

void f5(int cond, A *a, B *b) {
  bar(cond ? a : b);
}

void f6(int cond, id x, A *a) {
  bar(cond ? (id<P0, P1>) x : a);
}

void f7(int cond, id x, A *a) {
  bar(cond ? a : (id<P0, P1>) x);
}

void f8(int cond, id<P0,P1> x0, id<P0,P2> x1) {
  barP0(cond ? x0 : x1); // expected-warning {{incompatible operand types ('id<P0,P1>' and 'id<P0,P2>')}}
}

void f9(int cond, id<P0,P1> x0, id<P0,P2> x1) {
  barP1(cond ? x0 : x1); // expected-warning {{incompatible operand types ('id<P0,P1>' and 'id<P0,P2>')}}
}

void f10(int cond, id<P0,P1> x0, id<P0,P2> x1) {
  barP2(cond ? x0 : x1); // expected-warning {{incompatible operand types ('id<P0,P1>' and 'id<P0,P2>')}}
}

int f11(int cond, A* a, B* b) {
  return (cond? b : a)->x; // expected-error{{'A' does not have a member named 'x'}}
}
