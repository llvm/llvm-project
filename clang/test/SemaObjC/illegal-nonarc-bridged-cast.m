// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fblocks -verify %s
// rdar://10597832

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString
@end

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

id CreateSomething();
NSString *CreateNSString();

void from_cf() {
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning {{'__bridge_transfer' casts have no effect when not using ARC}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString(); // expected-warning {{'__bridge_transfer' casts have no effect when not using ARC}}
  (__bridge int*)CFCreateSomething();  // expected-warning {{'__bridge' casts have no effect when not using ARC}}  \
                                       // expected-warning {{expression result unused}}
  id obj3 = (__bridge id)CFGetSomething(); // expected-warning {{'__bridge' casts have no effect when not using ARC}}
  id obj4 = (__bridge NSString*)CFGetString(); // expected-warning {{'__bridge' casts have no effect when not using ARC}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning {{'__bridge_retained' casts have no effect when not using ARC}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning {{'__bridge_retained' casts have no effect when not using ARC}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-warning {{'__bridge' casts have no effect when not using ARC}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString(); // expected-warning {{'__bridge' casts have no effect when not using ARC}} 
}

void fixits() {
  id obj1 = (id)CFCreateSomething();
  CFTypeRef cf1 = (CFTypeRef)CreateSomething();
}

#pragma clang diagnostic ignored "-Warc-bridge-casts-disallowed-in-nonarc"

void to_cf_ignored(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // no-warning
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // no-warning
}
