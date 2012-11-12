// RUN: %clang_cc1 -Wno-objc-root-class -verify %s
// rdar://11273060

@interface I
- (void) setP : (int)arg;
@end

@interface J
  - (int) P;
@end

@interface K @end

@interface II @end

@implementation II
- (void) Meth : (I*) arg {
  arg.P++; // expected-error {{no getter method 'P' for increment of property}}
  --arg.P; // expected-error {{no getter method 'P' for decrement of property}}
}
- (void) Meth1 : (J*) arg {
  arg.P++; // expected-error {{no setter method 'setP:' for increment of property}}
  arg.P--; // expected-error {{no setter method 'setP:' for decrement of property}}
}

- (void) Meth2 : (K*) arg {
  arg.P++; // expected-error {{property 'P' not found on object of type 'K *'}}
  arg.P--; // expected-error {{property 'P' not found on object of type 'K *'}}
}
@end
