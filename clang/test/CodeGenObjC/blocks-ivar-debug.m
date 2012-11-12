// RUN: %clang_cc1 -g %s -fblocks -S -o %t
// Radar 7959934

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

