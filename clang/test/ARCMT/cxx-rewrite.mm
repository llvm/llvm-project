// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c++ %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c++ %s > %t
// RUN: diff %t %s.result
// DISABLE: mingw32

#include "Common.h"

@interface NSString : NSObject
+(id)string;
@end

struct foo {
    NSString *s;
    foo(NSString *s): s([s retain]){
        NSAutoreleasePool *pool = [NSAutoreleasePool new];
        [[[NSString string] retain] release];
        [pool drain];
        if (s)
          [s release];
    }
    ~foo(){ [s release]; }
private:
    foo(foo const &);
    foo &operator=(foo const &);
};

int main(){
    NSAutoreleasePool *pool = [NSAutoreleasePool new];

    foo f([[NSString string] autorelease]);

    [pool drain];
    return 0;
}
