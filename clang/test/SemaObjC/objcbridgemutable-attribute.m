// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://15498044

typedef struct __attribute__((objc_bridge_mutable(NSMutableDictionary))) __CFDictionary * CFMutableDictionaryRef; // expected-note {{declared here}}

typedef struct __attribute__((objc_bridge_mutable(12))) __CFDictionaryB1 * CFMutableDictionaryB1Ref; // expected-error {{parameter of 'objc_bridge_mutable' attribute must be a single name of an Objective-C class}}

typedef struct __attribute__((objc_bridge_mutable(P))) __CFDictionaryB2 * CFMutableDictionaryB2Ref; // expected-note {{declared here}}

typedef struct __attribute__((objc_bridge_mutable(NSMutableDictionary, Unknown))) __CFDictionaryB3 * CFMutableDictionaryB3Ref;  // expected-error {{use of undeclared identifier 'Unknown'}}

typedef struct __attribute__((objc_bridge_mutable)) __CFDictionaryB4 * CFMutableDictionaryB4Ref;  // expected-error {{'objc_bridge_mutable' attribute takes one argument}}

@interface NSDictionary
@end

@interface NSMutableDictionary : NSDictionary
@end

@protocol P @end

void Test(NSMutableDictionary *md, NSDictionary *nd, CFMutableDictionaryRef mcf, CFMutableDictionaryB2Ref bmcf) {

  (void) (CFMutableDictionaryRef)md;
  (void) (CFMutableDictionaryRef)nd; // expected-warning {{'NSDictionary' cannot bridge to 'CFMutableDictionaryRef' (aka 'struct __CFDictionary *')}}
  (void) (NSDictionary *)mcf; // ok, bridgt_type NSMutableDictionary can be type-cast to its super class.
  NSDictionary *nsdobj = (NSMutableDictionary*)0;
  (void) (NSMutableDictionary *)mcf; // ok;
  (void) (NSMutableDictionary *)bmcf; // expected-error {{CF object of type 'CFMutableDictionaryB2Ref' (aka 'struct __CFDictionaryB2 *') is bridged to 'P', which is not an Objective-C class}}
  
}

