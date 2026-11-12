// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

// Test that accessor stubs for default-synthesized ObjC accessors
// have a valid source location.

__attribute__((objc_root_class))
@interface NSObject
+ (id)alloc;
@end

@interface NSString : NSObject
@end

@interface MyData : NSObject
struct Data {
    NSString *name;
};
@property struct Data data;
@end
// CHECK: ObjCImplementationDecl {{.*}}line:[[@LINE+2]]{{.*}} MyData
// CHECK: ObjCMethodDecl {{.*}}col:23 implicit - setData: 'void'
@implementation MyData
@end
