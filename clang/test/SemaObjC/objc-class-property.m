// RUN: %clang_cc1 -fsyntax-only -verify %s

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
@property (class) int c2;
@property (class) int x;
@end

@implementation A
@dynamic x; // refers to the instance property
@dynamic (class) x; // refers to the class property
@synthesize z, c2; // expected-error {{@synthesize not allowed on a class property 'c2'}}
@dynamic c; // refers to the class property
@end

int test() {
  A *a = [[A alloc] init];
  return a.x + A.c;
}
