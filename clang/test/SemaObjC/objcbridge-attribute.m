// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://15454846

typedef struct __attribute__ ((objc_bridge(NSError))) __CFErrorRef * CFErrorRef; // expected-note 2 {{declared here}}

typedef struct __attribute__((objc_bridge(12))) __CFMyColor  *CFMyColorRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef struct __attribute__ ((objc_bridge)) __CFArray *CFArrayRef; // expected-error {{parameter of 'objc_bridge' attribute must be a single name of an Objective-C class}}

typedef void *  __attribute__ ((objc_bridge(NSURL))) CFURLRef;  // expected-error {{'objc_bridge' attribute only applies to struct or union}}

typedef void * CFStringRef __attribute__ ((objc_bridge(NSString))); // expected-error {{'objc_bridge' attribute only applies to struct or union}}

typedef struct __attribute__((objc_bridge(NSLocale, NSError))) __CFLocale *CFLocaleRef;// expected-error {{use of undeclared identifier 'NSError'}}

typedef struct __CFData __attribute__((objc_bridge(NSData))) CFDataRef; // expected-error {{'objc_bridge' attribute only applies to struct or union}}

typedef struct __attribute__((objc_bridge(NSDictionary))) __CFDictionary * CFDictionaryRef;

typedef struct __CFSetRef * CFSetRef __attribute__((objc_bridge(NSSet))); // expected-error {{'objc_bridge' attribute only applies to struct or union}};

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) * CFUColorRef; // expected-error {{'objc_bridge' attribute only applies to struct or union}};

typedef union __CFUColor __attribute__((objc_bridge(NSUColor))) *CFUColor1Ref; // expected-error {{'objc_bridge' attribute only applies to struct or union}};

typedef union __attribute__((objc_bridge(NSUColor))) __CFUPrimeColor XXX;
typedef XXX *CFUColor2Ref;

@interface I
{
   __attribute__((objc_bridge(NSError))) void * color; // expected-error {{'objc_bridge' attribute only applies to struct or union}};
}
@end

@protocol NSTesting @end
@class NSString;

typedef struct __attribute__((objc_bridge(NSTesting))) __CFError *CFTestingRef; // expected-note {{declared here}}

id Test1(CFTestingRef cf) {
  return (NSString *)cf; // expected-error {{CF object of type 'CFTestingRef' (aka 'struct __CFError *') is bridged to 'NSTesting', which is not an Objective-C class}}
}

typedef CFErrorRef CFErrorRef1;

typedef CFErrorRef1 CFErrorRef2;

@interface NSError @end

@interface MyError : NSError
@end

@interface NSUColor @end

@class NSString;

void Test2(CFErrorRef2 cf, NSError *ns, NSString *str, Class c, CFUColor2Ref cf2) {
  (void)(NSString *)cf; // expected-warning {{'CFErrorRef2' (aka 'struct __CFErrorRef *') bridges to NSError, not 'NSString'}}
  (void)(NSError *)cf; // okay
  (void)(MyError*)cf; // okay,
  (void)(NSUColor *)cf2; // okay
  (void)(CFErrorRef)ns; // okay
  (void)(CFErrorRef)str;  // expected-warning {{'NSString' cannot bridge to 'CFErrorRef' (aka 'struct __CFErrorRef *')}}
  (void)(Class)cf; // expected-warning {{'CFErrorRef2' (aka 'struct __CFErrorRef *') bridges to NSError, not 'Class'}}
  (void)(CFErrorRef)c; // expected-warning {{'Class' cannot bridge to 'CFErrorRef'}}
}
