// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result
// DISABLE: mingw32

#include "Common.h"

@interface A : NSObject {
@package
    id object;
}
@end

@interface B : NSObject {
  id _prop;
  xpc_object_t _xpc_prop;
}
- (BOOL)containsSelf:(A*)a;
@property (retain) id prop;
@property (retain) xpc_object_t xpc_prop;
@end

@implementation A
@end

@implementation B
- (BOOL)containsSelf:(A*)a {
    return a->object == self;
}

-(id) prop {
  return _prop;
}
-(void) setProp:(id) newVal {
  [_prop autorelease];
  _prop = [newVal retain];
}
-(void) setProp2:(CFTypeRef) newVal {
  [_prop autorelease];
  _prop = (id)CFRetain(newVal);
}

-(id) xpc_prop {
  return _xpc_prop;
}
-(void) setXpc_prop:(xpc_object_t) newVal {
  [_xpc_prop autorelease];
  _xpc_prop = xpc_retain(newVal);
}
@end

void NSLog(id, ...);

int main (int argc, const char * argv[]) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    A *a = [[A new] autorelease];
    B *b = [[B new] autorelease];
    NSLog(@"%s", [b containsSelf:a] ? "YES" : "NO");
    [pool drain];
    return 0;
}

void test(A *prevVal, A *newVal) {
  [prevVal autorelease];
  prevVal = [newVal retain];
}
