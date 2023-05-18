// RUN: %clang_cc1 -fsyntax-only -verify %s

#define PLACE_IN_TCB(NAME) __attribute__((enforce_tcb(NAME)))
#define PLACE_IN_TCB_LEAF(NAME) __attribute__((enforce_tcb_leaf(NAME)))

__attribute__((objc_root_class))
@interface AClass
@property(readonly) id propertyNotInAnyTCB;
@end

@implementation AClass
- (void)inTCBFoo PLACE_IN_TCB("foo") {
  [self notInAnyTCB]; // expected-warning{{calling 'notInAnyTCB' is a violation of trusted computing base 'foo'}}
}
- (void)inTCBFooAsLeaf PLACE_IN_TCB_LEAF("foo") {
  [self notInAnyTCB]; // no-warning
}
- (void)notInAnyTCB {
}
+ (void)classMethodNotInAnyTCB {
}
+ (void)classMethodInTCBFoo PLACE_IN_TCB("foo") {
  [self inTCBFoo];       // no-warning
  [self inTCBFooAsLeaf]; // no-warning
  [self notInAnyTCB];    // expected-warning{{calling 'notInAnyTCB' is a violation of trusted computing base 'foo'}}
}
@end

PLACE_IN_TCB("foo")
void call_objc_method(AClass *object) {
  [object inTCBFoo];                // no-warning
  [object inTCBFooAsLeaf];          // no-warning
  [object notInAnyTCB];             // expected-warning{{calling 'notInAnyTCB' is a violation of trusted computing base 'foo'}}
  [AClass classMethodNotInAnyTCB];  // expected-warning{{calling 'classMethodNotInAnyTCB' is a violation of trusted computing base 'foo'}}
  [AClass classMethodInTCBFoo];     // no-warning
  (void)object.propertyNotInAnyTCB; // expected-warning{{calling 'propertyNotInAnyTCB' is a violation of trusted computing base 'foo'}}
}
