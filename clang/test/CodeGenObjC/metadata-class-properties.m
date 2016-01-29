// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck -check-prefix=CHECK-FRAGILE %s

// CHECK: @"\01l_OBJC_$_CLASS_PROP_LIST_Foo_$_Category" = private global {{.*}} section "__DATA, __objc_const", align 8
// CHECK: @"\01l_OBJC_$_CATEGORY_Foo_$_Category" = private global %struct._category_t { {{.*}} @"\01l_OBJC_$_CLASS_PROP_LIST_Foo_$_Category" {{.*}} }, section "__DATA, __objc_const", align 8

// CHECK: !{i32 1, !"Objective-C Class Properties", i32 64}

// CHECK-FRAGILE: @"\01l_OBJC_$_CLASS_PROP_LIST_Foo_Category" = private global {{.*}} section "__OBJC,__property,regular,no_dead_strip", align 8
// CHECK-FRAGILE: @OBJC_CATEGORY_Foo_Category = private global %struct._objc_category { {{.*}}, i32 64, {{.*}} @"\01l_OBJC_$_CLASS_PROP_LIST_Foo_Category" {{.*}} }, section "__OBJC,__category,regular,no_dead_strip", align 8

// CHECK-FRAGILE: !{i32 1, !"Objective-C Class Properties", i32 64}

@interface Foo @end

@protocol Proto
@property (class, readonly) int proto_property;
@end

@interface Foo (Category) <Proto> @end

@implementation Foo (Category)
+(int)proto_property { return 0; }
@end
