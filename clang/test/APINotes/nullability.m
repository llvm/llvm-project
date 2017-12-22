// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules  -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -Wno-private-module -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

// Test with Swift version 3.0. This should only affect the few APIs that have an entry in the 3.0 tables.

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -Wno-private-module -fapinotes-swift-version=3.0 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify -DSWIFT_VERSION_3_0 -fmodules-ignore-macro=SWIFT_VERSION_3_0

#import <SomeKit/SomeKit.h>

int main() {
  A *a;

#if SWIFT_VERSION_3_0
  float *fp =  // expected-warning{{incompatible pointer types initializing 'float *' with an expression of type 'A * _Nullable'}}
    [a transform: 0 integer: 0];
#else
  float *fp =  // expected-warning{{incompatible pointer types initializing 'float *' with an expression of type 'A *'}}
    [a transform: 0 integer: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
#endif

  [a setNonnullAInstance: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [A setNonnullAInstance: 0]; // no warning
  
  [a setNonnullAClass: 0]; // no warning
  [A setNonnullAClass: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  [a setNonnullABoth: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [A setNonnullABoth: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  [a setInternalProperty: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

#if SWIFT_VERSION_3_0
  // Version 3 information overrides header information.
  [a setExplicitNonnullInstance: 0]; //  okay
  [a setExplicitNullableInstance: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
#else
  // Header information overrides unversioned information.
  [a setExplicitNonnullInstance: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [a setExplicitNullableInstance: 0]; // okay
#endif

  return 0;
}

