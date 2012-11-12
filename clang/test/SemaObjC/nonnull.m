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

void func6();
void func7();

void
foo (int i1, int i2, int i3, void (^cp1)(), void (^cp2)(), void (^cp3)())
{
  func1(cp1, cp2, i1);
  
  func1(0, cp2, i1);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  func1(cp1, 0, i1);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  func1(cp1, cp2, 0);
  
  
  func3(0, i2, cp3, i3); // expected-warning {{null passed to a callee which requires a non-null argument}}
  func3(cp3, i2, 0, i3);  // expected-warning {{null passed to a callee which requires a non-null argument}}
  
  func4(0, cp1); // expected-warning {{null passed to a callee which requires a non-null argument}}
  func4(cp1, 0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  
  // Shouldn't these emit warnings?  Clang doesn't, and neither does GCC.  It
  // seems that the checking should handle Objective-C pointers.
  func6((NSObject*) 0); // no-warning
  func7((NSObject*) 0); // no-warning
}

void func5(int) NONNULL_ATTR; //  no warning

// rdar://6857843
struct dispatch_object_s {
    int x;
};

typedef union {
    long first;
    struct dispatch_object_s *_do;
} dispatch_object_t __attribute__((transparent_union));

__attribute__((nonnull))
void _dispatch_queue_push_list(dispatch_object_t _head); // no warning

void func6(dispatch_object_t _head) {
  _dispatch_queue_push_list(0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  _dispatch_queue_push_list(_head._do);  // no warning
}

// rdar://9287695
#define NULL (void*)0

@interface NSObject 
- (void)doSomethingWithNonNullPointer:(void *)ptr :(int)iarg : (void*)ptr1 __attribute__((nonnull(1, 3)));
+ (void)doSomethingClassyWithNonNullPointer:(void *)ptr __attribute__((nonnull(1)));
@end

extern void DoSomethingNotNull(void *db) __attribute__((nonnull(1)));

@interface IMP 
{
  void * vp;
}
@end

@implementation IMP
- (void) Meth {
  NSObject *object;
  [object doSomethingWithNonNullPointer:NULL:1:NULL]; // expected-warning 2 {{null passed to a callee which requires a non-null argument}}
  [object doSomethingWithNonNullPointer:vp:1:NULL]; // expected-warning {{null passed to a callee which requires a non-null argument}}
  [NSObject doSomethingClassyWithNonNullPointer:NULL]; // expected-warning {{null passed to a callee which requires a non-null argument}}
  DoSomethingNotNull(NULL); // expected-warning {{null passed to a callee which requires a non-null argument}}
  [object doSomethingWithNonNullPointer:vp:1:vp];
}
@end

