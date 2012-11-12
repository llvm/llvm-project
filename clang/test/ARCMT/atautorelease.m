// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result
// DISABLE: mingw32

#include "Common.h"

void NSLog(id, ...);

int main (int argc, const char * argv[]) {

    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    if (argc) {
        NSAutoreleasePool * pool = [NSAutoreleasePool  new];
        NSLog(@"%s", "YES");
        [pool drain];
    }
    [pool drain];

    NSAutoreleasePool * pool1 = [[NSAutoreleasePool alloc] init];
    NSLog(@"%s", "YES");
    [pool1 release];

    return 0;
}

void f(void) {
  NSAutoreleasePool *pool1;

  pool1 = [NSAutoreleasePool new];
  int x = 4;

  NSAutoreleasePool *pool2 = [[NSAutoreleasePool alloc] init];
  ++x;
  [pool2 drain];

  [pool1 release];
}

int UIApplicationMain(int argc, char *argv[]);

int main2(int argc, char *argv[]) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    int result = UIApplicationMain(argc, argv);
    [pool release];
    return result;
}

@interface Foo : NSObject
@property (assign) id myProp;
@end

@implementation Foo
@synthesize myProp;

-(void)test:(id)p {
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  [pool drain];
  self.myProp = p;
}
@end
