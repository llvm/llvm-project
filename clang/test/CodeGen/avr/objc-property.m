// RUN: %clang_cc1 -triple avr -emit-llvm -fobjc-runtime=macosx %s -o /dev/null

__attribute__((objc_root_class))
@interface Foo

@property(strong) Foo *f;

@end

@implementation Foo

@synthesize f = _f;

@end
