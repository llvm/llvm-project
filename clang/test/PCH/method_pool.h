/* For use with the method_pool.m test */

/* Whitespace below is significant */











@interface TestMethodPool1
+ alloc;
- (double)instMethod:(int)foo;
@end

@interface TestMethodPool2
- (char)instMethod:(int)foo;
@end

@implementation TestMethodPool1
+ alloc { return 0; }

- (double)instMethod:(int)foo {
  return foo;
}
@end

@implementation TestMethodPool2
- (char)instMethod:(int)foo {
  return foo;
}
@end
