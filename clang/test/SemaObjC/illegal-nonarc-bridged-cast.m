// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fblocks -Wcast-qual -verify %s

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;
@class NSString;

@interface NSString
@end

CFTypeRef CFCreateSomething(void);
CFStringRef CFCreateString(void);
CFTypeRef CFGetSomething(void);
CFStringRef CFGetString(void);

id CreateSomething(void);
NSString *CreateNSString(void);

void from_cf(void) {
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning {{'__bridge_transfer' casts have no effect when not using ARC}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString(); // expected-warning {{'__bridge_transfer' casts have no effect when not using ARC}}
  (__bridge int*)CFCreateSomething(); // expected-warning {{expression result unused}} expected-warning {{cast from 'const void *' to 'int *' drops const qualifier}}
  id obj3 = (__bridge id)CFGetSomething();
  id obj4 = (__bridge NSString*)CFGetString();
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning {{'__bridge_retained' casts have no effect when not using ARC}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning {{'__bridge_retained' casts have no effect when not using ARC}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething();
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();
}

void fixits(void) {
  id obj1 = (id)CFCreateSomething();
  CFTypeRef cf1 = (CFTypeRef)CreateSomething();
}

#pragma clang diagnostic ignored "-Warc-bridge-casts-disallowed-in-nonarc"

void to_cf_ignored(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // no-warning
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // no-warning
}

// Check that clang doesn't warn about dropping const from Objective-C object
// types.
void test_wcast_qual(void) {
  CFStringRef c;
  NSString *n0 = (NSString *)c;
  NSString **n1 = (NSString **)&c;
  const NSString *n2;
  const NSString **n3;
  void *p0 = (void *)n2;
  void **p1 = (void **)n3;
}
