// RUN: %clang_cc1 -fblocks -emit-llvm %s -o /dev/null
// rdar://6394879

@interface bork
- (id)B:(void (^)(void))blk;
- (void)C;
@end
@implementation bork
- (id)B:(void (^)(void))blk {
  __attribute__((__blocks__(byref))) bork* new = ((void *)0);
  blk();
}
- (void)C {
  __attribute__((__blocks__(byref))) id var;
  [self B:^(void) {}];
}
@end
