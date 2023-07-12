// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef const void *CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);
typedef const struct __CFString *CFStringRef;

@interface NSString
@end

CFTypeRef CFCreateSomething(void);
CFStringRef CFCreateString(void);
CFTypeRef CFGetSomething(void);
CFStringRef CFGetString(void);

id CreateSomething(void);
NSString *CreateNSString(void);

void from_cf(void) {
  id obj1 = (__bridge_transfer id)CFCreateSomething();
  id obj2 = (__bridge_transfer NSString*)CFCreateString();
  (__bridge int*)CFCreateSomething(); // expected-error{{incompatible types casting 'CFTypeRef' (aka 'const void *') to 'int *' with a __bridge cast}}
  id obj3 = (__bridge id)CFGetSomething();
  id obj4 = (__bridge NSString*)CFGetString();
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething();
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString();
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething();
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();

  CFTypeRef cf5 = (__bridge_retain CFTypeRef)CreateSomething(); // expected-error {{unknown cast annotation __bridge_retain; did you mean __bridge_retained?}}
  // CHECK: fix-it:"{{.*}}":{34:20-34:35}:"__bridge_retained"
}

CFTypeRef fixits(void) {
  id obj1 = (id)CFCreateSomething(); // expected-error{{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}
  // CHECK: fix-it:"{{.*}}":{39:17-39:17}:"CFBridgingRelease("
  // CHECK: fix-it:"{{.*}}":{39:36-39:36}:")"

  CFTypeRef cf1 = (CFTypeRef)CreateSomething(); // expected-error{{cast of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{44:30-44:30}:"CFBridgingRetain("
  // CHECK: fix-it:"{{.*}}":{44:47-44:47}:")"

  return (obj1); // expected-error{{implicit conversion of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{50:10-50:10}:"(__bridge CFTypeRef)"
  // CHECK: fix-it:"{{.*}}":{50:10-50:10}:"CFBridgingRetain"
}

CFTypeRef fixitsWithSpace(id obj) {
  return(obj); // expected-error{{implicit conversion of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{58:9-58:9}:"(__bridge CFTypeRef)"
  // CHECK: fix-it:"{{.*}}":{58:9-58:9}:" CFBridgingRetain"
}

typedef const struct __attribute__((objc_bridge(id))) __CFAnnotatedObject *CFAnnotatedObjectRef;
CFAnnotatedObjectRef CFGetAnnotated(void);

void testObjCBridgeId(void) {
  id obj;
  obj = (__bridge id)CFGetAnnotated();
  obj = (__bridge NSString*)CFGetAnnotated();
  obj = (__bridge_transfer id)CFGetAnnotated();
  obj = (__bridge_transfer NSString*)CFGetAnnotated();

  CFAnnotatedObjectRef ref;
  ref = (__bridge CFAnnotatedObjectRef) CreateSomething();
  ref = (__bridge CFAnnotatedObjectRef) CreateNSString();
  ref = (__bridge_retained CFAnnotatedObjectRef) CreateSomething();
  ref = (__bridge_retained CFAnnotatedObjectRef) CreateNSString();
}

typedef const struct __attribute__((objc_bridge(UIFont))) __CTFont * CTFontRef;

id testObjCBridgeUnknownTypeToId(CTFontRef font) {
  id x = (__bridge id)font;
  return x;
}

