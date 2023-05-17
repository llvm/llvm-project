// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -Wundeclared-selector -Wno-int-conversion -fixit %t
// RUN: %clang_cc1 -x objective-c -Wundeclared-selector -Wno-int-conversion -Werror %t
// rdar://14039037

@interface NSObject @end

@interface LogoutController : NSObject
- (void)close;
- (void)closed;
- (void) open : (id) file_id;
@end

@implementation LogoutController

- (void)close  { }
- (void)closed  { }

- (SEL)Meth
{
  return @selector(cloze);
}
- (void) open : (id) file_id {}

- (SEL)Meth1
{
  return @selector(ope:);
}

@end

// rdar://7853549
@interface rdar7853549 : NSObject
- (int) bounds;
@end

@implementation rdar7853549
- (int) bounds { return 0; }
- (void)PrivateMeth { int bounds = [self bonds]; }
- (void)OtherPrivateMeth : (id) p { int bounds = [p bonds]; }
@end
