// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc-only -fblocks  -emit-llvm -o - %s | FileCheck %s
// rdar://10150823

@interface Test {
@package
    Test ** __strong objects;
}
@end

id newObject(void);
void runWithBlock(void(^)(int i));

@implementation Test

- (void)testWithObjectInBlock {
    Test **children = objects;
    runWithBlock(^(int i){
        children[i] = newObject();
    });
}

@end
// CHECK: call ptr @objc_assign_strongCast
// CHECK: call ptr @objc_assign_strongCast

