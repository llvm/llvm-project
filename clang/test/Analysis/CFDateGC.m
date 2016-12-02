// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -verify -fobjc-gc %s  -Wno-implicit-function-declaration

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h and CoreFoundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including [Core]Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef const void * CFTypeRef;
void CFRelease(CFTypeRef cf);
CFTypeRef CFRetain(CFTypeRef cf);
CFTypeRef CFMakeCollectable(CFTypeRef cf);
typedef const struct __CFAllocator * CFAllocatorRef;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
extern CFAbsoluteTime CFDateGetAbsoluteTime(CFDateRef theDate);
typedef struct objc_object {} *id;
typedef signed char BOOL;
static __inline__ __attribute__((always_inline)) id NSMakeCollectable(CFTypeRef cf) { return 0; }
@protocol NSObject  - (BOOL)isEqual:(id)object;
- (oneway void)release;
- (id)retain;
@end
@class NSArray;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

CFAbsoluteTime CFAbsoluteTimeGetCurrent();

CFAbsoluteTime f1_use_after_release() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  [NSMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released}}
  return t;
}

// The following two test cases verifies that CFMakeCollectable is a no-op
// in non-GC mode and a "release" in GC mode.
CFAbsoluteTime f2_use_after_release() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  [(id) CFMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released}}
  return t;
}

CFAbsoluteTime f2_noleak() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  [(id) CFMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  t = CFDateGetAbsoluteTime(date);  // no-warning
  CFRelease(date); // no-warning
  return t;
}

void f3_leak_with_gc() {
  CFDateRef date = CFDateCreate(0, CFAbsoluteTimeGetCurrent()); // expected-warning 2 {{leak}}
  [[(id) date retain] release];
}

// The following test case verifies that we "stop tracking" a retained object
// when it is passed as an argument to an implicitly defined function.
CFAbsoluteTime f4() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  CFRetain(date);
  some_implicitly_defined_function_stop_tracking(date); // no-warning
  return t;
}
