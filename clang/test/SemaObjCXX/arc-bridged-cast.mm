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
