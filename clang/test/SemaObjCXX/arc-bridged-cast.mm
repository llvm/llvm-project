// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify %s

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

template<typename IdType, typename StringType, typename IntPtrType>
void from_cf() {
  id obj1 = (__bridge_transfer IdType)CFCreateSomething();
  id obj2 = (__bridge_transfer StringType)CFCreateString();
  (__bridge IntPtrType)CFCreateSomething(); // expected-error{{incompatible types casting 'CFTypeRef' (aka 'const void *') to 'int *' with a __bridge cast}}
  id obj3 = (__bridge IdType)CFGetSomething();
  id obj4 = (__bridge StringType)CFGetString();
}

template void from_cf<id, NSString*, int*>(); // expected-note{{in instantiation of function template specialization}}

template<typename IdType, typename StringType>
void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained IdType)CreateSomething();
  CFStringRef cf2 = (__bridge_retained StringType)CreateNSString();
  CFTypeRef cf3 = (__bridge IdType)CreateSomething();
  CFStringRef cf4 = (__bridge StringType)CreateNSString();
}

template void to_cf<CFTypeRef, CFStringRef>(id);

typedef const struct __attribute__((objc_bridge(id))) __CFAnnotatedObject *CFAnnotatedObjectRef;
CFAnnotatedObjectRef CFGetAnnotated();

void testObjCBridgeId() {
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

struct __CFAnnotatedObject {
} cf0;

extern const CFAnnotatedObjectRef r0;
extern const CFAnnotatedObjectRef r1 = &cf0;
extern "C" const CFAnnotatedObjectRef r2;
extern "C" const CFAnnotatedObjectRef r3 = &cf0;

void testExternC() {
  id obj;
  obj = (id)r0;
  obj = (id)r1; // expected-error{{cast of C pointer type 'CFAnnotatedObjectRef' (aka 'const struct __CFAnnotatedObject *') to Objective-C pointer type 'id' requires a bridged cast}} expected-note{{use __bridge to convert directly}} expected-note{{use __bridge_transfer to transfer ownership of a +1 'CFAnnotatedObjectRef'}}
  obj = (id)r2;
  obj = (id)r3; // expected-error{{cast of C pointer type 'CFAnnotatedObjectRef' (aka 'const struct __CFAnnotatedObject *') to Objective-C pointer type 'id' requires a bridged cast}} expected-note{{use __bridge to convert directly}} expected-note{{use __bridge_transfer to transfer ownership of a +1 'CFAnnotatedObjectRef'}}
}
