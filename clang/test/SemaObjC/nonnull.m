#include "nonnull.h"

// RUN: %clang_cc1 -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// REQUIRES: LP64
@class NSObject;

NONNULL_ATTR
int f1(int x); //  no warning
int f2(int *x) __attribute__ ((nonnull (1)));
int f3(int *x) __attribute__ ((nonnull (0))); // expected-error {{'nonnull' attribute parameter 1 is out of bounds}}
int f4(int *x, int *y) __attribute__ ((nonnull (1,2)));
int f5(int *x, int *y) __attribute__ ((nonnull (2,1)));
int f6(NSObject *x) __attribute__ ((nonnull (1))); // no-warning
int f7(NSObject *x) __attribute__ ((nonnull)); // no-warning


extern void func1 (void (^block1)(), void (^block2)(), int) __attribute__((nonnull));

extern void func3 (void (^block1)(), int, void (^block2)(), int)
__attribute__((nonnull(1,3)));

extern void func4 (void (^block1)(), void (^block2)()) __attribute__((nonnull(1)))
__attribute__((nonnull(2)));

void func6(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C23, conflicting with a subsequent definition}}
void func7();

void
foo (int i1, int i2, int i3, void (^cp1)(), void (^cp2)(), void (^cp3)())
{
  func1(cp1, cp2, i1);
  
  func1(0, cp2, i1);  // expected-warning {{null passed to a callee that requires a non-null argument}}
  func1(cp1, 0, i1);  // expected-warning {{null passed to a callee that requires a non-null argument}}
  func1(cp1, cp2, 0);
  
  
  func3(0, i2, cp3, i3); // expected-warning {{null passed to a callee that requires a non-null argument}}
  func3(cp3, i2, 0, i3);  // expected-warning {{null passed to a callee that requires a non-null argument}}
  
  func4(0, cp1); // expected-warning {{null passed to a callee that requires a non-null argument}}
  func4(cp1, 0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  
  func6((NSObject*) 0); // expected-warning {{passing arguments to 'func6' without a prototype is deprecated in all versions of C and is not supported in C23}}
  func7((NSObject*) 0); // expected-warning {{passing arguments to 'func7' without a prototype is deprecated in all versions of C and is not supported in C23}}
}

void func5(int) NONNULL_ATTR; //  no warning

struct dispatch_object_s {
    int x;
};

typedef union {
    long first;
    struct dispatch_object_s *_do;
} dispatch_object_t __attribute__((transparent_union));

__attribute__((nonnull))
void _dispatch_queue_push_list(dispatch_object_t _head); // no warning

void func6(dispatch_object_t _head) { // expected-note {{conflicting prototype is here}}
  _dispatch_queue_push_list(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  _dispatch_queue_push_list(_head._do);  // no warning
}

#define NULL (void*)0

@interface NSObject 
- (void)doSomethingWithNonNullPointer:(void *)ptr :(int)iarg : (void*)ptr1 __attribute__((nonnull(1, 3)));
+ (void)doSomethingClassyWithNonNullPointer:(void *)ptr __attribute__((nonnull(1)));
- (void*)returnsCNonNull __attribute__((returns_nonnull)); // no-warning
- (id)returnsObjCNonNull __attribute__((returns_nonnull)); // no-warning
- (int)returnsIntNonNull __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to return values that are pointers}}
@end

extern void DoSomethingNotNull(void *db) __attribute__((nonnull(1)));

@interface IMP 
{
  void * vp;
}
- (void*) testRetNull __attribute__((returns_nonnull));
@end

@implementation IMP
- (void) Meth {
  NSObject *object;
  [object doSomethingWithNonNullPointer:NULL:1:NULL]; // expected-warning 2 {{null passed to a callee that requires a non-null argument}}
  [object doSomethingWithNonNullPointer:vp:1:NULL]; // expected-warning {{null passed to a callee that requires a non-null argument}}
  [NSObject doSomethingClassyWithNonNullPointer:NULL]; // expected-warning {{null passed to a callee that requires a non-null argument}}
  DoSomethingNotNull(NULL); // expected-warning {{null passed to a callee that requires a non-null argument}}
  [object doSomethingWithNonNullPointer:vp:1:vp];
}
- (void*) testRetNull {
  return 0; // expected-warning {{null returned from method that requires a non-null return value}}
}
@end

__attribute__((objc_root_class))
@interface TestNonNullParameters
- (void) doNotPassNullParameterNonPointerArg:(int)__attribute__((nonnull))x; // expected-warning {{'nonnull' attribute only applies to pointer arguments}}
- (void) doNotPassNullParameter:(id)__attribute__((nonnull))x;
- (void) doNotPassNullParameterArgIndex:(id)__attribute__((nonnull(1)))x; // expected-warning {{'nonnull' attribute when used on parameters takes no arguments}}
- (void) doNotPassNullOnMethod:(id)x __attribute__((nonnull(1)));
@end

void test(TestNonNullParameters *f) {
  [f doNotPassNullParameter:0]; // expected-warning {{null passed to a callee that requires a non-null argument}}
  [f doNotPassNullParameterArgIndex:0]; // no-warning
  [f doNotPassNullOnMethod:0]; // expected-warning {{null passed to a callee that requires a non-null argument}}
}


void PR18795(int (^g)(const char *h, ...) __attribute__((nonnull(1))) __attribute__((nonnull))) {
  g(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}
void PR18795_helper() {
  PR18795(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

void (^PR23117)(int *) = ^(int *p1) __attribute__((nonnull(1))) {};

typedef int *intptr;
#pragma clang assume_nonnull begin
intptr a, b;
intptr c, (*d)();
#pragma clang assume_nonnull end
