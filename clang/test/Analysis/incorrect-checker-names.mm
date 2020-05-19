// RUN: %clang_analyze_cc1 -fblocks -verify %s -Wno-objc-root-class \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=nullability \
// RUN:   -analyzer-checker=osx

#include "Inputs/system-header-simulator-for-nullability.h"
#include "os_object_base.h"

struct OSIterator : public OSObject {
  static const OSMetaClass * const metaClass;
};

@interface TestObject : NSObject
- (int *_Nonnull)returnsNonnull;
- (int *_Nullable)returnsNullable;
- (int *)returnsUnspecified;
- (void)takesNonnull:(int *_Nonnull)p;
- (void)takesNullable:(int *_Nullable)p;
- (void)takesUnspecified:(int *)p;
@property(readonly, strong) NSString *stuff;
@end

TestObject * getUnspecifiedTestObject();
TestObject *_Nonnull getNonnullTestObject();
TestObject *_Nullable getNullableTestObject();

int getRandom();

typedef struct Dummy { int val; } Dummy;

void takesNullable(Dummy *_Nullable);
void takesNonnull(Dummy *_Nonnull);
void takesUnspecified(Dummy *);

Dummy *_Nullable returnsNullable();
Dummy *_Nonnull returnsNonnull();
Dummy *returnsUnspecified();
int *_Nullable returnsNullableInt();

template <typename T> T *eraseNullab(T *p) { return p; }

void takesAttrNonnull(Dummy *p) __attribute((nonnull(1)));

void testBasicRules() {
  // FIXME: None of these should be tied to a modeling checker.
  Dummy *p = returnsNullable();
  int *ptr = returnsNullableInt();
  // Make every dereference a different path to avoid sinks after errors.
  switch (getRandom()) {
  case 0: {
    Dummy &r = *p; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 1: {
    int b = p->val; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 2: {
    int stuff = *ptr; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  } break;
  case 3:
    takesNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter [nullability.NullablePassedToNonnull]}}
    break;
  case 4: {
    Dummy d;
    takesNullable(&d);
    Dummy dd(d);
    break;
  }
  case 5:
    takesAttrNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null [nullability.NullableDereferenced]}}
    break;
  default: { Dummy d = *p; } break; // expected-warning {{Nullable pointer is dereferenced [nullability.NullableDereferenced]}}
  }
  if (p) {
    takesNonnull(p);
    if (getRandom()) {
      Dummy &r = *p;
    } else {
      int b = p->val;
    }
  }
  Dummy *q = 0;
  if (getRandom()) {
    takesNullable(q);
  // FIXME: This shouldn't be tied to a modeling checker.
    takesNonnull(q); // expected-warning {{Null passed to a callee that requires a non-null 1st parameter [nullability.NullPassedToNonnull]}}
  }
  Dummy a;
  Dummy *_Nonnull nonnull = &a;
  // FIXME: This shouldn't be tied to a modeling checker.
  nonnull = q; // expected-warning {{Null assigned to a pointer which is expected to have non-null value [nullability.NullPassedToNonnull]}}
  q = &a;
  takesNullable(q);
  takesNonnull(q);
}

typedef int NSInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@class NSDictionary;
@interface NSError : NSObject <NSCopying, NSCoding> {}
+ (id)errorWithDomain:(NSString *)domain code:(NSInteger)code userInfo:(NSDictionary *)dict;
@end

struct __CFError {};
typedef struct __CFError* CFErrorRef;

void foo(CFErrorRef* error) { // expected-warning{{Function accepting CFErrorRef* should have a non-void return value to indicate whether or not an error occurred [osx.coreFoundation.CFError]}}
  // FIXME: This shouldn't be tied to a modeling checker.
  *error = 0;  // expected-warning {{Potential null dereference.  According to coding standards documented in CoreFoundation/CFError.h the parameter may be null [osx.NSOrCFErrorDerefChecker]}}
}

bool write_into_out_param_on_success(OS_RETURNS_RETAINED OSObject **obj);

void use_out_param_leak() {
  OSObject *obj;
  // FIXME: This shouldn't be tied to a modeling checker.
  write_into_out_param_on_success(&obj); // expected-warning{{Potential leak of an object stored into 'obj' [osx.cocoa.RetainCountBase]}}
}
