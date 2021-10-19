// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 -fobjc-gc-only %s
// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 -fobjc-gc-only -x objective-c++ %s

#define CF_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
typedef unsigned NSUInteger;
typedef const void * CFTypeRef;
CFTypeRef CFMakeCollectable(CFTypeRef cf) CF_AUTOMATED_REFCOUNT_UNAVAILABLE; // expected-note {{unavailable}}
void *__strong NSAllocateCollectable(NSUInteger size, NSUInteger options);

void test1(CFTypeRef *cft) {
  CFTypeRef c = CFMakeCollectable(cft); // expected-error {{CFMakeCollectable will leak the object that it receives in ARC}} \
                // expected-error {{unavailable}}
  NSAllocateCollectable(100, 0); // expected-error {{call returns pointer to GC managed memory; it will become unmanaged in ARC}}
}

@interface I1 {
  __strong void *gcVar; // expected-error {{GC managed memory will become unmanaged in ARC}}
}
@end;
