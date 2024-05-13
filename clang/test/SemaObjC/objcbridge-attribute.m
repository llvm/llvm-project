// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct __attribute__ ((objc_bridge(NSError))) __CFErrorRef * CFErrorRef; // expected-note 3 {{declared here}}

typedef struct __attribute__ ((objc_bridge(MyError))) __CFMyErrorRef * CFMyErrorRef; // expected-note 1 {{declared here}}

typedef struct __attribute__((objc_bridge(12))) __CFMyColor  *CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct __attribute__ ((objc_bridge)) __CFArray *CFArrayRef; // expected-error {{'objc_bridge' attribute takes one argument}}

typedef void *  __attribute__ ((objc_bridge(NSURL))) CFURLRef;  // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef void * CFStringRef __attribute__ ((objc_bridge(NSString))); // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef struct __attribute__((objc_bridge(NSLocale, NSError))) __CFLocale *CFLocaleRef;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct __CFData __attribute__((objc_bridge(NSData))) CFDataRef; // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef struct __attribute__((objc_bridge(NSDictionary))) __CFDictionary * CFDictionaryRef; // expected-note {{declared here}}

typedef struct __CFSetRef * CFSetRef __attribute__((objc_bridge(NSSet))); // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) * CFUColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) *CFUColor1Ref; // expected-error {{parameter of 'objc_bridge' attribute must be 'id' when used on a typedef}}

typedef union __attribute__((objc_bridge(NSUColor))) __CFUPrimeColor XXX;
typedef XXX *CFUColor2Ref;

typedef const void *ConstVoidRef __attribute__((objc_bridge(id)));
typedef void *VoidRef __attribute__((objc_bridge(id)));
typedef struct Opaque *OpaqueRef __attribute__((objc_bridge(id))); // expected-error {{'objc_bridge(id)' is only allowed on structs and typedefs of void pointers}}

#if !__has_feature(objc_bridge_id_on_typedefs)
#error objc_bridge(id) on typedefs feature not found!
#endif

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute only applies to structs, unions, classes, and typedefs}};
}
@end

@protocol NSTesting @end
@class NSString;

typedef struct __attribute__((objc_bridge(NSTesting))) __CFError *CFTestingRef; // expected-note {{declared here}}

id Test1(CFTestingRef cf) {
  return (NSString *)cf; // expected-error {{CF object of type 'CFTestingRef' (aka 'struct __CFError *') is bridged to 'NSTesting', which is not an Objective-C class}}
}

typedef CFErrorRef CFErrorRef1;

typedef CFErrorRef1 CFErrorRef2; // expected-note {{declared here}}

@protocol P1 @end
@protocol P2 @end
@protocol P3 @end
@protocol P4 @end
@protocol P5 @end

@interface NSError<P1, P2, P3> @end // expected-note 3 {{declared here}}

@interface MyError : NSError // expected-note 1 {{declared here}}
@end

@interface NSUColor @end

@class NSString;

void Test2(CFErrorRef2 cf, NSError *ns, NSString *str, Class c, CFUColor2Ref cf2) {
  (void)(NSString *)cf; // expected-warning {{'CFErrorRef2' (aka 'struct __CFErrorRef *') bridges to NSError, not 'NSString'}}
  (void)(NSError *)cf; // okay
  (void)(MyError*)cf; // expected-warning {{'CFErrorRef2' (aka 'struct __CFErrorRef *') bridges to NSError, not 'MyError'}} 
  (void)(NSUColor *)cf2; // okay
  (void)(CFErrorRef)ns; // okay
  (void)(CFErrorRef)str;  // expected-warning {{'NSString' cannot bridge to 'CFErrorRef' (aka 'struct __CFErrorRef *')}}
  (void)(Class)cf; // expected-warning {{'CFErrorRef2' (aka 'struct __CFErrorRef *') bridges to NSError, not 'Class'}}
  (void)(CFErrorRef)c; // expected-warning {{'Class' cannot bridge to 'CFErrorRef'}}
}


void Test3(CFErrorRef cf, NSError *ns) {
  (void)(id)cf; // okay
 (void)(id<P1, P2>)cf; // okay
 (void)(id<P1, P2, P4>)cf; // expected-warning {{'CFErrorRef' (aka 'struct __CFErrorRef *') bridges to NSError, not 'id<P1,P2,P4>'}}
}

void Test4(CFMyErrorRef cf) {
   (void)(id)cf; // okay
 (void)(id<P1, P2>)cf; // ok
 (void)(id<P1, P2, P3>)cf; // ok
 (void)(id<P2, P3>)cf; // ok
 (void)(id<P1, P2, P4>)cf; // expected-warning {{'CFMyErrorRef' (aka 'struct __CFMyErrorRef *') bridges to MyError, not 'id<P1,P2,P4>'}}
}

void Test5(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {
 (void)(CFErrorRef)ID; // ok
 (void)(CFErrorRef)P123; // ok
 (void)(CFErrorRef)P1234; // ok
 (void)(CFErrorRef)P12; // ok 
 (void)(CFErrorRef)P23; // ok
}

void Test6(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {

 (void)(CFMyErrorRef)ID; // ok
 (void)(CFMyErrorRef)P123; // ok
 (void)(CFMyErrorRef)P1234; // ok
 (void)(CFMyErrorRef)P12; // ok
 (void)(CFMyErrorRef)P23; // ok
}

typedef struct __attribute__ ((objc_bridge(MyPersonalError))) __CFMyPersonalErrorRef * CFMyPersonalErrorRef;  // expected-note 1 {{declared here}}

@interface MyPersonalError : NSError <P4> // expected-note 1 {{declared here}}
@end

void Test7(id<P1, P2, P3> P123, id ID, id<P1, P2, P3, P4> P1234, id<P1, P2> P12, id<P2, P3> P23) {
 (void)(CFMyPersonalErrorRef)ID; // ok
 (void)(CFMyPersonalErrorRef)P123; // ok
 (void)(CFMyPersonalErrorRef)P1234; // ok
 (void)(CFMyPersonalErrorRef)P12; // ok
 (void)(CFMyPersonalErrorRef)P23; // ok
}

void Test8(CFMyPersonalErrorRef cf) {
  (void)(id)cf; // ok
  (void)(id<P1>)cf; // ok
  (void)(id<P1, P2>)cf; // ok
  (void)(id<P1, P2, P3>)cf; // ok
  (void)(id<P1, P2, P3, P4>)cf; // ok
  (void)(id<P1, P2, P3, P4, P5>)cf; // expected-warning {{'CFMyPersonalErrorRef' (aka 'struct __CFMyPersonalErrorRef *') bridges to MyPersonalError, not 'id<P1,P2,P3,P4,P5>'}}
}

CFDictionaryRef bar(void) __attribute__((cf_returns_not_retained));
@class NSNumber;

void Test9(void) {
  NSNumber *w2 = (NSNumber*) bar(); // expected-error {{CF object of type 'CFDictionaryRef' (aka 'struct __CFDictionary *') is bridged to 'NSDictionary', which is not an Objective-C class}}
}

@interface NSObject @end

@interface NSFont : NSObject @end

typedef struct __attribute__ ((objc_bridge(NSFont))) __CFFontRef * CFFontRef;

void Test10(CFFontRef cf) {
  (void)(__bridge NSObject *)cf;
}
