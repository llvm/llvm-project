// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -emit-llvm -o /dev/null %s

typedef struct
{
  int i;
} GAXBackboardState  __attribute__ ((aligned(32))); // minimum alignment is 32-byte boundary

@interface GAXSpringboard @end

@implementation GAXSpringboard
{
 GAXBackboardState _reflectedBackboardState;
}

- (void) MyMethod
{
 GAXBackboardState newBackboardState;
 ^{
    _reflectedBackboardState = newBackboardState;
    return newBackboardState.i;
  }();
}
@end

