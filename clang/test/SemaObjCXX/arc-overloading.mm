// RUN: %clang_cc1 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify -fblocks -Wno-objc-root-class %s

// Simple ownership conversions + diagnostics.
int &f0(id __strong const *); // expected-note{{candidate function not viable: 1st argument ('__weak id *') has __weak ownership, but parameter has __strong ownership}}

void test_f0() {
  id __strong *sip;
  id __strong const *csip;
  id __weak *wip;
  id __autoreleasing *aip;
  id __unsafe_unretained *uip;

  int &ir1 = f0(sip);
  int &ir2 = f0(csip);
  int &ir3 = f0(aip);
  int &ir4 = f0(uip);
  f0(wip); // expected-error{{no matching function for call to 'f0'}}
}

// Simple overloading
int &f1(id __strong const *);
float &f1(id __weak const *);

void test_f1() {
  id __strong *sip;
  id __strong const *csip;
  id __weak *wip;
  id __autoreleasing *aip;
  id __unsafe_unretained *uip;

  int &ir1 = f1(sip);
  int &ir2 = f1(csip);
  float &fr1 = f1(wip);
  int &ir3 = f1(aip);
  int &ir4 = f1(uip);
}

// Simple overloading
int &f2(id __strong const *); // expected-note{{candidate function}}
float &f2(id __autoreleasing const *); // expected-note{{candidate function}}

void test_f2() {
  id __strong *sip;
  id __strong const *csip;
  id __weak *wip;
  id __autoreleasing *aip;
  id __unsafe_unretained *uip;

  // Prefer non-ownership conversions to ownership conversions.
  int &ir1 = f2(sip);
  int &ir2 = f2(csip);
  float &fr1 = f2(aip);

  f2(uip); // expected-error{{call to 'f2' is ambiguous}}
}

// Writeback conversion
int &f3(id __autoreleasing *); // expected-note{{candidate function not viable: 1st argument ('__unsafe_unretained id *') has __unsafe_unretained ownership, but parameter has __autoreleasing ownership}}

void test_f3() {
  id __strong sip;
  id __weak wip;
  id __autoreleasing aip;
  id __unsafe_unretained uip;

  int &ir1 = f3(&sip);
  int &ir2 = f3(&wip);
  int &ir3 = f3(&aip);
  f3(&uip); // expected-error{{no matching function for call to 'f3'}}
}

// Writeback conversion vs. no conversion
int &f4(id __autoreleasing *);
float &f4(id __strong *);

void test_f4() {
  id __strong sip;
  id __weak wip;
  id __autoreleasing aip;
  extern __weak id weak_global_ptr;

  float &fr1 = f4(&sip);
  int &ir1 = f4(&wip);
  int &ir2 = f4(&aip);
  int &ir3 = f4(&weak_global_ptr); // expected-error{{passing address of non-local object to __autoreleasing parameter for write-back}}
}

// Writeback conversion vs. other conversion.
int &f5(id __autoreleasing *);
float &f5(id const __unsafe_unretained *);

void test_f5() {
  id __strong sip;
  id __weak wip;
  id __autoreleasing aip;

  int &ir1 = f5(&wip);
  float &fr1 = f5(&sip);
  int &ir2 = f5(&aip);
}

@interface A
@end

int &f6(id __autoreleasing *);
float &f6(id const __unsafe_unretained *);

void test_f6() {
  A* __strong sip;
  A* __weak wip;
  A* __autoreleasing aip;

  int &ir1 = f6(&wip);
  float &fr1 = f6(&sip);
  int &ir2 = f6(&aip);
}

// Reference binding
void f7(__strong id&); // expected-note{{candidate function not viable: 1st argument ('__weak id') has __weak ownership, but parameter has __strong ownership}} \
 // expected-note{{candidate function not viable: 1st argument ('__autoreleasing id') has __autoreleasing ownership, but parameter has __strong ownership}} \
 // expected-note{{candidate function not viable: 1st argument ('__unsafe_unretained id') has __unsafe_unretained ownership, but parameter has __strong ownership}}

void test_f7() {
  __strong id strong_id;
  __weak id weak_id;
  __autoreleasing id autoreleasing_id;
  __unsafe_unretained id unsafe_id;
  f7(strong_id);
  f7(weak_id); // expected-error{{no matching function for call to 'f7'}}
  f7(autoreleasing_id); // expected-error{{no matching function for call to 'f7'}}
  f7(unsafe_id); // expected-error{{no matching function for call to 'f7'}}
}

void f8(const __strong id&);

void test_f8() {
  __strong id strong_id;
  __weak id weak_id;
  __autoreleasing id autoreleasing_id;
  __unsafe_unretained id unsafe_id;

  f8(strong_id);
  f8(weak_id);
  f8(autoreleasing_id);
  f8(unsafe_id);
}

int &f9(__strong id&);
float &f9(const __autoreleasing id&);

void test_f9() {
  __strong id strong_id;
  __weak id weak_id;
  __autoreleasing id autoreleasing_id;
  __unsafe_unretained id unsafe_id;

  int &ir1 = f9(strong_id);
  float &fr1 = f9(autoreleasing_id);
  float &fr2 = f9(unsafe_id);
  float &fr2a = f9(weak_id);

  __strong A *strong_a;
  __weak A *weak_a;
  __autoreleasing A *autoreleasing_a;
  __unsafe_unretained A *unsafe_unretained_a;
  float &fr3 = f9(strong_a);
  float &fr4 = f9(autoreleasing_a);
  float &fr5 = f9(unsafe_unretained_a);
  float &fr6 = f9(weak_a);

  const __autoreleasing id& ar1 = strong_a;
  const __autoreleasing id& ar2 = autoreleasing_a;
  const __autoreleasing id& ar3 = unsafe_unretained_a;
  const __autoreleasing id& ar4 = weak_a;
}

// rdar://9790531
void f9790531(void *inClientData); // expected-note {{candidate function not viable: cannot implicitly convert argument of type 'MixerEQGraphTestDelegate *const __strong' to 'void *' for 1st argument under ARC}}
void f9790531_1(struct S*inClientData); // expected-note {{candidate function not viable}}
void f9790531_2(char * inClientData); // expected-note {{candidate function not viable}}

@class UIApplication;

@interface MixerEQGraphTestDelegate
- (void)applicationDidFinishLaunching;
@end

@implementation MixerEQGraphTestDelegate
- (void)applicationDidFinishLaunching {
    f9790531(self); // expected-error {{no matching function for call to 'f9790531'}}
    f9790531_1(self); // expected-error {{no matching function for call to 'f9790531_1'}}
    f9790531_2(self); // expected-error {{no matching function for call to 'f9790531_2'}}
}
@end

class rdar10142572 {
  id f() __attribute__((ns_returns_retained));
  id g(); // expected-note{{previous declaration}}
};

id rdar10142572::f() { return 0; } // okay: merged down
id __attribute__((ns_returns_retained)) rdar10142572::g() { return 0; } // expected-error{{function declared with 'ns_returns_retained' attribute was previously declared without the 'ns_returns_retained' attribute}}
