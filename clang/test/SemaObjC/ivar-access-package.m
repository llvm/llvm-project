// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef unsigned char BOOL;

@interface NSObject {
  id isa;
}
+new;
+alloc;
-init;
-autorelease;
@end

@interface NSAutoreleasePool : NSObject
- drain;
@end
 
@interface A : NSObject {
@package
    id object;
}
@end

@interface B : NSObject
- (BOOL)containsSelf:(A*)a;
@end

@implementation A
@end

@implementation B
- (BOOL)containsSelf:(A*)a {
    return a->object == self;
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

