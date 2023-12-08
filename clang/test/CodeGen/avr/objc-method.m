// RUN: %clang_cc1 -triple avr -emit-llvm -fobjc-runtime=macosx %s -o /dev/null

__attribute__((objc_root_class))
@interface Foo

- (id)foo;
- (id)bar;

@end

@implementation Foo

- (id)foo {
  return self;
}

- (id)bar {
  return [self foo];
}

@end
