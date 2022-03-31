// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
// rdar://8528170

@interface NSObject @end

@protocol MyProtocol
- (int) level;
- (void) setLevel:(int)inLevel;
@end

@interface MyClass : NSObject <MyProtocol>
@end

int main (void)
{
    id<MyProtocol> c;
    c.level = 10;
    return 0;
}
