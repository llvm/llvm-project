// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions -Wno-objc-root-class %s

@class A, B, C;

void test1() {
  goto L; // expected-error{{cannot jump}}
  goto L2; // expected-error{{cannot jump}}
  goto L3; // expected-error{{cannot jump}}
  @try {   // expected-note {{jump bypasses initialization of @try block}}
L: ;
  } @catch (A *x) { // expected-note {{jump bypasses initialization of @catch block}}
L2: ;
  } @catch (B *x) {
  } @catch (C *c) {
  } @finally {// expected-note {{jump bypasses initialization of @finally block}}
L3: ;
  }
  
  @try {
    goto L4; // expected-error{{cannot jump}}
    goto L5; // expected-error{{cannot jump}}
  } @catch (C *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L5: ;
    goto L6; // expected-error{{cannot jump}}
  } @catch (B *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L6: ;
  } @finally { // expected-note {{jump bypasses initialization of @finally block}}
  L4: ;
  }
 
  
  @try { // expected-note 2 {{jump bypasses initialization of @try block}}
  L7: ;
  } @catch (C *c) {
    goto L7; // expected-error{{cannot jump}}
  } @finally {
    goto L7; // expected-error{{cannot jump}}
  }
  
  goto L8;  // expected-error{{cannot jump}}
  @try { 
  } @catch (A *c) {
  } @catch (B *c) {
  } @catch (C *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L8: ;
  }
  
  // rdar://6810106
  id X;
  goto L9;    // expected-error{{cannot jump}}
  goto L10;   // ok
  @synchronized    // expected-note {{jump bypasses initialization of @synchronized block}}
  ( ({ L10: ; X; })) {
  L9:
    ;
  }
}

void test2(int a) {
  if (a) goto L0;
  @try {} @finally {}
 L0:
  return;
}

// rdar://6803963
void test3() {
  @try {
    goto blargh;
  blargh: ;
  } @catch (...) {}
}

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello {

  @try {
    goto blargh;     // expected-error {{cannot jump}}
  } @catch (...) {   // expected-note {{jump bypasses initialization of @catch block}}
  blargh: ;
  }
}

+ (void)meth2 {
    int n; void *P;
    goto L0;     // expected-error {{cannot jump}}
    typedef int A[n];  // expected-note {{jump bypasses initialization of VLA typedef}}
  L0:
    
    goto L1;      // expected-error {{cannot jump}}
    A b, c[10];        // expected-note 2 {{jump bypasses initialization of variable length array}}
  L1:
    goto L2;     // expected-error {{cannot jump}}
    A d[n];      // expected-note {{jump bypasses initialization of variable length array}}
  L2:
    return;
}

@end
