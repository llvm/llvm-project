// RUN: %clang_cc1 -debug-info-kind=limited %s -fblocks -S -o %t

@interface NSObject {
  struct objc_object *isa;
}
@end
@interface Foo : NSObject {
  int _prop;
}
@end

@implementation Foo
- (int)doSomething {
  int (^blk)(void) = ^{ return _prop; };
  return blk();
}

@end

