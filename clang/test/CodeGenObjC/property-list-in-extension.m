// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

// Checks metadata for properties in a few cases.


// Property from a class extension:
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
// CHECK: = private global [12 x i8] c"Ti,V_myprop\00",
// CHECK: @"\01l_OBJC_$_PROP_LIST_Foo" = private global { i32, i32, [1 x %struct._prop_t] }

// Readonly property in interface made readwrite in a category:
@interface FooRO
@property (readonly) int evolvingprop;
@end

@interface FooRO ()
@property int evolvingprop;
@end

@implementation FooRO
@synthesize evolvingprop = _evolvingprop;
@end
// Metadata for _evolvingprop should be present, and PROP_LIST for FooRO should
// still have only one entry, and the one entry should point to the version of
// the property with a getter and setter.
// CHECK: [[getter:@OBJC_PROP_NAME_ATTR[^ ]+]] = private global [13 x i8] c"evolvingprop\00"
// CHECK: [[setter:@OBJC_PROP_NAME_ATTR[^ ]+]] = private global [18 x i8] c"Ti,V_evolvingprop\00",
// CHECK: @"\01l_OBJC_$_PROP_LIST_FooRO" = private global { i32, i32, [1 x %struct._prop_t] }{{.*}}[[getter]]{{.*}}[[setter]]
