// RUN: %clang_cc1 -verify -fobjc-exceptions %s
@interface A @end
@protocol P;

void f(void) {
  @try {
  } @catch (void a) { // expected-error{{@catch parameter is not a pointer to an interface type}}
  } @catch (int) { // expected-error{{@catch parameter is not a pointer to an interface type}}
  } @catch (int *b) { // expected-error{{@catch parameter is not a pointer to an interface type}}
  } @catch (id <P> c) { // expected-error{{illegal qualifiers on @catch parameter}}
  } @catch(A* a) { }
}

