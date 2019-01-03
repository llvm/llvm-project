// RUN: %clang_analyze_cc1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s
// RUN: %clang_analyze_cc1 -DEXTRA=1 -DAP1=1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s
// RUN: %clang_analyze_cc1 -DEXTRA=1 -DAP2=1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s
// RUN: %clang_analyze_cc1 -DEXTRA=1 -DAP3=1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s
// RUN: %clang_analyze_cc1 -DEXTRA=1 -DAP4=1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s
// RUN: %clang_analyze_cc1 -DEXTRA=1 -DAP5=1 -fobjc-arc -triple x86_64-darwin\
// RUN:   -analyzer-checker=core,osx.cocoa.RunLoopAutoreleaseLeak -verify %s

#include "../Inputs/system-header-simulator-for-objc-dealloc.h"

#ifndef EXTRA

void just_runloop() { // No warning: no statements in between
  @autoreleasepool {
    [[NSRunLoop mainRunLoop] run]; // no-warning
  }
}

void just_xpcmain() { // No warning: no statements in between
  @autoreleasepool {
    xpc_main(); // no-warning
  }
}

void runloop_init_before() { // Warning: object created before the loop.
  @autoreleasepool {
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool followed by the launch of main run loop may never get released; consider moving them to a separate autorelease pool}}
    (void) object;
    [[NSRunLoop mainRunLoop] run]; 
  }
}

void runloop_init_before_separate_pool() { // No warning: separate autorelease pool.
  @autoreleasepool {
    NSObject *object;
    @autoreleasepool {
      object = [[NSObject alloc] init]; // no-warning
    }
    (void) object;
    [[NSRunLoop mainRunLoop] run]; 
  }
}

void xpcmain_init_before() { // Warning: object created before the loop.
  @autoreleasepool {
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool followed by the launch of xpc_main may never get released; consider moving them to a separate autorelease pool}}
    (void) object;
    xpc_main(); 
  }
}

void runloop_init_before_two_objects() { // Warning: object created before the loop.
  @autoreleasepool {
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool followed by the launch of main run loop may never get released; consider moving them to a separate autorelease pool}}
    NSObject *object2 = [[NSObject alloc] init]; // no-warning, warning on the first one is enough.
    (void) object;
    (void) object2;
    [[NSRunLoop mainRunLoop] run];
  }
}

void runloop_no_autoreleasepool() {
  NSObject *object = [[NSObject alloc] init]; // no-warning
  (void)object;
  [[NSRunLoop mainRunLoop] run];
}

void runloop_init_after() { // No warning: objects created after the loop
  @autoreleasepool {
    [[NSRunLoop mainRunLoop] run]; 
    NSObject *object = [[NSObject alloc] init]; // no-warning
    (void) object;
  }
}

void no_crash_on_empty_children() {
  @autoreleasepool {
    for (;;) {}
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool followed by the launch of main run loop may never get released; consider moving them to a separate autorelease pool}}
    [[NSRunLoop mainRunLoop] run];
    (void) object;
  }
}

#endif

#ifdef AP1
int main() {
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool of last resort followed by the launch of main run loop may never get released; consider moving them to a separate autorelease pool}}
    (void) object;
    [[NSRunLoop mainRunLoop] run]; 
    return 0;
}
#endif

#ifdef AP2
// expected-no-diagnostics
int main() {
  NSObject *object = [[NSObject alloc] init]; // no-warning
  (void) object;
  @autoreleasepool {
    [[NSRunLoop mainRunLoop] run]; 
  }
  return 0;
}
#endif

#ifdef AP3
// expected-no-diagnostics
int main() {
    [[NSRunLoop mainRunLoop] run];
    NSObject *object = [[NSObject alloc] init]; // no-warning
    (void) object;
    return 0;
}
#endif

#ifdef AP4
int main() {
    NSObject *object = [[NSObject alloc] init]; // expected-warning{{Temporary objects allocated in the autorelease pool of last resort followed by the launch of xpc_main may never get released; consider moving them to a separate autorelease pool}}
    (void) object;
    xpc_main();
    return 0;
}
#endif

#ifdef AP5
@class NSString;
@class NSConstantString;
#define CF_BRIDGED_TYPE(T)    __attribute__((objc_bridge(T)))
typedef const CF_BRIDGED_TYPE(id) void * CFTypeRef;
typedef const struct CF_BRIDGED_TYPE(NSString) __CFString * CFStringRef;

typedef enum { WARNING } Level;
id do_log(Level, const char *);
#define log(level, msg) __extension__({ (do_log(level, msg)); })

@interface I
- foo;
@end

CFStringRef processString(const __NSConstantString *, void *);

#define CFSTR __builtin___CFStringMakeConstantString

int main() {
  I *i;
  @autoreleasepool {
    NSString *s1 = (__bridge_transfer NSString *)processString(0, 0);
    NSString *s2 = (__bridge_transfer NSString *)processString((CFSTR("")), ((void *)0));
    log(WARNING, "Hello world!");
  }
  [[NSRunLoop mainRunLoop] run];
  [i foo]; // no-crash // expected-warning{{Temporary objects allocated in the autorelease pool of last resort followed by the launch of main run loop may never get released; consider moving them to a separate autorelease pool}}
}
#endif
