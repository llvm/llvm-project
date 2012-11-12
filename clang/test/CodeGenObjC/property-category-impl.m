// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s

// rdar : // 8093297

@interface Foo @end

@protocol Proto
@property (readonly) int proto_property;
@end

@interface Foo (Category) <Proto> @end

@implementation Foo (Category)
-(int)proto_property { return 0; }
@end


// CHECK: l_OBJC_$_PROP_LIST_Foo_$_Category" = internal global
// CHECK: l_OBJC_$_CATEGORY_Foo_$_Category" = internal global
// CHECK: l_OBJC_$_PROP_LIST_Foo_$_Category
