// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef const void *CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);
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

  // rdar://problem/9629566 - temporary workaround
  CFTypeRef cf5 = (__bridge_retain CFTypeRef)CreateSomething(); // expected-error {{unknown cast annotation __bridge_retain; did you mean __bridge_retained?}}
  // CHECK: fix-it:"{{.*}}":{35:20-35:35}:"__bridge_retained"
}

CFTypeRef fixits() {
  id obj1 = (id)CFCreateSomething(); // expected-error{{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}
  // CHECK: fix-it:"{{.*}}":{40:17-40:17}:"CFBridgingRelease("
  // CHECK: fix-it:"{{.*}}":{40:36-40:36}:")"

  CFTypeRef cf1 = (CFTypeRef)CreateSomething(); // expected-error{{cast of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{45:30-45:30}:"CFBridgingRetain("
  // CHECK: fix-it:"{{.*}}":{45:47-45:47}:")"

  return (obj1); // expected-error{{implicit conversion of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{51:10-51:10}:"(__bridge CFTypeRef)"
  // CHECK: fix-it:"{{.*}}":{51:10-51:10}:"CFBridgingRetain"
}

CFTypeRef fixitsWithSpace(id obj) {
  return(obj); // expected-error{{implicit conversion of Objective-C pointer type 'id' to C pointer type 'CFTypeRef' (aka 'const void *') requires a bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFTypeRef' (aka 'const void *')}}
  // CHECK: fix-it:"{{.*}}":{59:9-59:9}:"(__bridge CFTypeRef)"
  // CHECK: fix-it:"{{.*}}":{59:9-59:9}:" CFBridgingRetain"
}
