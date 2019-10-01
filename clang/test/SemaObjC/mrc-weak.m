// RUN: %clang_cc1 -fobjc-runtime-has-weak -fobjc-weak -fsyntax-only -verify %s

__attribute__((objc_root_class))
@interface A
@property (weak) id wa; // expected-note {{property declared here}}
@property (weak) id wb;
@property (weak) id wc; // expected-note {{property declared here}}
@property (weak) id wd;
@property (unsafe_unretained) id ua;
@property (unsafe_unretained) id ub; // expected-note {{property declared here}}
@property (unsafe_unretained) id uc;
@property (unsafe_unretained) id ud;
@property (strong) id sa;
@property (strong) id sb; // expected-note {{property declared here}}
@property (strong) id sc;
@property (strong) id sd;
@end

@implementation A {
  id _wa; // expected-error {{existing instance variable '_wa' for __weak property 'wa' must be __weak}}
  __weak id _wb;
  __unsafe_unretained id _wc; // expected-error {{existing instance variable '_wc' for __weak property 'wc' must be __weak}}
  id _ua;
  __weak id _ub; // expected-error {{existing instance variable '_ub' for property 'ub' with unsafe_unretained attribute must be __unsafe_unretained}}
  __unsafe_unretained id _uc;
  id _sa;
  __weak id _sb; // expected-error {{existing instance variable '_sb' for strong property 'sb' may not be __weak}}
  __unsafe_unretained id _sc;
}
@synthesize wa = _wa; // expected-note {{property synthesized here}}
@synthesize wb = _wb;
@synthesize wc = _wc; // expected-note {{property synthesized here}}
@synthesize wd = _wd;
@synthesize ua = _ua;
@synthesize ub = _ub; // expected-note {{property synthesized here}}
@synthesize uc = _uc;
@synthesize ud = _ud;
@synthesize sa = _sa;
@synthesize sb = _sb; // expected-note {{property synthesized here}}
@synthesize sc = _sc;
@synthesize sd = _sd;
@end

void test_goto() {
  goto after; // expected-error {{cannot jump from this goto statement to its label}}
  __weak id x; // expected-note {{jump bypasses initialization of __weak variable}}}
after:
  return;
}

void test_weak_cast(id *value) {
  __weak id *a = (__weak id*) value;
  id *b = (__weak id*) value; // expected-error {{initializing 'id *' with an expression of type '__weak id *' changes retain/release properties of pointer}}
  __weak id *c = (id*) value; // expected-error {{initializing '__weak id *' with an expression of type 'id *' changes retain/release properties of pointer}}
}

void test_unsafe_unretained_cast(id *value) {
  __unsafe_unretained id *a = (__unsafe_unretained id*) value;
  id *b = (__unsafe_unretained id*) value;
  __unsafe_unretained id *c = (id*) value;
}

void test_cast_qualifier_inference(__weak id *value) {
  __weak id *a = (id*) value;
  __unsafe_unretained id *b = (id *)value; // expected-error {{initializing 'id *' with an expression of type '__weak id *' changes retain/release properties of pointer}}
}

