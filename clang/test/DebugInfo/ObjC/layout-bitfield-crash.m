// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-gc -emit-llvm -debug-info-kind=limited -o - %s
// Check that this doesn't crash when compiled with debugging on.
@class Foo;
typedef struct Bar *BarRef;

@interface Baz
@end

@interface Foo
- (void) setFlag;
@end

@implementation Baz

- (void) a:(BarRef)b
{
  Foo* view = (Foo*)self;
  [view setFlag];
}

@end


@implementation Foo
{
  int flag : 1;
}

- (void) setFlag
{
  if (!flag)
    flag = 1;
}

@end
