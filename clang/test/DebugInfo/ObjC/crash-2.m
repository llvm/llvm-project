// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -debug-info-kind=limited -S %s -o -
// REQUIRES: x86-registered-target

@class Bar;
@interface Foo
@property (strong, nonatomic) Bar *window;
@end

@interface Foo__ : Foo
@end
@implementation Foo__
@end

@implementation Foo
@synthesize window = _window;
@end

