// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-weak -fobjc-runtime-has-weak -emit-llvm %s -o - | FileCheck %s

// Checks metadata for properties in a few cases.


// Property from a class extension:
__attribute__((objc_root_class))
@interface Foo
@end

@interface Foo()
@property int myprop;
@end

@implementation Foo
@synthesize myprop = _myprop;
@end
// Metadata for _myprop should be present, and PROP_LIST for Foo should have
// only one entry.
// CHECK: = private unnamed_addr constant [12 x i8] c"Ti,V_myprop\00",
// CHECK: @"\01l_OBJC_$_PROP_LIST_Foo" = private global { i32, i32, [1 x %struct._prop_t] }

// Readonly property in interface made readwrite in a category:
__attribute__((objc_root_class))
@interface FooRO
@property (readonly) int evolvingprop;
@property (nonatomic,readonly,getter=isBooleanProp) int booleanProp;
@property (nonatomic,readonly,weak) Foo *weakProp;
@end

@interface FooRO ()
@property int evolvingprop;
@property int booleanProp;
@property Foo *weakProp;
@end

@implementation FooRO
@synthesize evolvingprop = _evolvingprop;
@end
// Metadata for _evolvingprop should be present, and PROP_LIST for FooRO should
// still have only one entry, and the one entry should point to the version of
// the property with a getter and setter.
// CHECK: [[evolvinggetter:@OBJC_PROP_NAME_ATTR[^ ]+]] = private unnamed_addr constant [13 x i8] c"evolvingprop\00"
// CHECK: [[evolvingsetter:@OBJC_PROP_NAME_ATTR[^ ]+]] = private unnamed_addr constant [18 x i8] c"Ti,V_evolvingprop\00",
// CHECK: [[booleanmetadata:@OBJC_PROP_NAME_ATTR[^ ]+]] = private unnamed_addr constant [34 x i8] c"Ti,N,GisBooleanProp,V_booleanProp\00"
// CHECK: [[weakmetadata:@OBJC_PROP_NAME_ATTR[^ ]+]] = private unnamed_addr constant [23 x i8] c"T@\22Foo\22,W,N,V_weakProp\00"
// CHECK: @"\01l_OBJC_$_PROP_LIST_FooRO" = private global { i32, i32, [3 x %struct._prop_t] }{{.*}}[[evolvinggetter]]{{.*}}[[evolvingsetter]]{{.*}}[[booleanmetadata]]
