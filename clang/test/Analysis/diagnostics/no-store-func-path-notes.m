// RUN: %clang_analyze_cc1 -x objective-c -analyzer-checker=core,nullability -analyzer-output=text -Wno-objc-root-class -fblocks -verify %s

#include "../Inputs/system-header-simulator-for-nullability.h"

extern int coin();

@interface I : NSObject
- (int)initVar:(int *)var param:(int)param;
@end

@implementation I
- (int)initVar:(int *)var param:(int)param {
  if (param) { // expected-note{{'param' is 0}}
               // expected-note@-1{{Taking false branch}}
    *var = 1;
    return 0;
  }
  return 1; // expected-note{{Returning without writing to '*var'}}
}
@end

int foo(I *i) {
  int x;                            //expected-note{{'x' declared without an initial value}}
  int out = [i initVar:&x param:0]; //expected-note{{Calling 'initVar:param:'}}
                                    //expected-note@-1{{Returning from 'initVar:param:'}}
  if (out)                          //expected-note{{'out' is 1}}
                                    //expected-note@-1{{Taking true branch}}
    return x;                       //expected-warning{{Undefined or garbage value returned to caller}}
                                    //expected-note@-1{{Undefined or garbage value returned to caller}}
  return 0;
}

int initializer1(int *p, int x) {
  if (x) { // expected-note{{'x' is 0}}
           // expected-note@-1{{Taking false branch}}
    *p = 1;
    return 0;
  } else {
    return 1; // expected-note {{Returning without writing to '*p'}}
  }
}

int initFromBlock() {
  __block int z;
  ^{                     // expected-note {{Calling anonymous block}}
    int p;               // expected-note{{'p' declared without an initial value}}
    initializer1(&p, 0); // expected-note{{Calling 'initializer1'}}
                         // expected-note@-1{{Returning from 'initializer1'}}
    z = p;               // expected-warning{{Assigned value is garbage or undefined}}
                         // expected-note@-1{{Assigned value is garbage or undefined}}
  }();
  return z;
}

extern void expectNonNull(NSString * _Nonnull a);

@interface A : NSObject
- (void) initAMaybe;
@end

@implementation A {
  NSString * a;
}

- (void) initAMaybe {
  if (coin()) // expected-note{{Assuming the condition is false}}
              // expected-note@-1{{Taking false branch}}
    a = @"string";
} // expected-note{{Returning without writing to 'self->a'}}

- (void) passNullToNonnull {
  a = nil; // expected-note{{nil object reference stored to 'a'}}
  [self initAMaybe]; // expected-note{{Calling 'initAMaybe'}}
                     // expected-note@-1{{Returning from 'initAMaybe'}}
  expectNonNull(a); // expected-warning{{nil passed to a callee that requires a non-null 1st parameter}}
                    // expected-note@-1{{nil passed to a callee that requires a non-null 1st parameter}}
}

- (void) initAMaybeWithExplicitSelf {
  if (coin()) // expected-note{{Assuming the condition is false}}
              // expected-note@-1{{Taking false branch}}
    self->a = @"string";
} // expected-note{{Returning without writing to 'self->a'}}

- (void) passNullToNonnullWithExplicitSelf {
  self->a = nil; // expected-note{{nil object reference stored to 'a'}}
  [self initAMaybeWithExplicitSelf]; // expected-note{{Calling 'initAMaybeWithExplicitSelf'}}
                     // expected-note@-1{{Returning from 'initAMaybeWithExplicitSelf'}}
  expectNonNull(a); // expected-warning{{nil passed to a callee that requires a non-null 1st parameter}}
                    // expected-note@-1{{nil passed to a callee that requires a non-null 1st parameter}}
}

- (void) initPassedAMaybe:(A *) param {
  if (coin()) // expected-note{{Assuming the condition is false}}
              // expected-note@-1{{Taking false branch}}
    param->a = @"string";
} // expected-note{{Returning without writing to 'param->a'}}

- (void) useInitPassedAMaybe:(A *) paramA {
  paramA->a = nil; // expected-note{{nil object reference stored to 'a'}}
  [self initPassedAMaybe:paramA]; // expected-note{{Calling 'initPassedAMaybe:'}}
                                  // expected-note@-1{{Returning from 'initPassedAMaybe:'}}
  expectNonNull(paramA->a); // expected-warning{{nil passed to a callee that requires a non-null 1st parameter}}
                            // expected-note@-1{{nil passed to a callee that requires a non-null 1st parameter}}

}

@end
