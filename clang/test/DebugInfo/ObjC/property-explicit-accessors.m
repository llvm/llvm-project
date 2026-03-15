// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
 
// CHECK: !DIObjCProperty(name: "baseInt"
// CHECK-SAME:            setter: "mySetBaseInt:"
// CHECK-SAME:            getter: "myGetBaseInt"
// CHECK-SAME:            attributes: 2446
// CHECK-SAME:            type: ![[P1_TYPE:[0-9]+]]
//
// CHECK: ![[P1_TYPE]] = !DIBasicType(name: "int"

@interface BaseClass2 
{
	int _baseInt;
}
- (int) myGetBaseInt;
- (void) mySetBaseInt: (int) in_int;
@property(getter=myGetBaseInt,setter=mySetBaseInt:) int baseInt;
@end

@implementation BaseClass2

- (int) myGetBaseInt
{
        return _baseInt;
}

- (void) mySetBaseInt: (int) in_int
{
    _baseInt = 2 * in_int;
}
@end


void foo(BaseClass2 *ptr) {}
