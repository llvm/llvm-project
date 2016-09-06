// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface Root
-(id) alloc;
-(id) init;
@end

@interface A : Root {
  int x;
  int z;
}
@property int x;
@property int y;
@property int z;
@property(readonly) int ro, ro2;
@property (class) int c;
@end

@implementation A
@dynamic x;
@synthesize z;
@dynamic c;
@end

int test() {
  A *a = [[A alloc] init];
  return a.x;
}
