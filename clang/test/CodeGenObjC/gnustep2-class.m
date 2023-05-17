// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

@interface Super @end

@interface X : Super
{
	int ivar1;
	id ivar2;
}
@property (readonly) int x;
@property id y;
@end

@implementation X
@synthesize y;

- (int)x { return 12; }
+ (int)clsMeth { return 42; }
- (id)meth { return ivar2; }
@end

// Check that we get an ivar offset variable for the synthesised ivar.
// CHECK: @"__objc_ivar_offset_X.y.\01" = hidden global i32 16
//
// Check that we get a sensible metaclass method list.
// CHECK: internal global { ptr, i32, i64, [1 x { ptr, ptr, ptr }] }
// CHECK-SAME: @_c_X__clsMeth

// Check that we get a metaclass and that it is not an exposed symbol:
// CHECK: @._OBJC_METACLASS_X = internal global

// Check that we get a reference to the superclass symbol:
// CHECK: @._OBJC_CLASS_Super = external global ptr

// Check that we get an ivar list with all three ivars, in the correct order
// CHECK: private global { i32, i64, [3 x { ptr, ptr, ptr, i32, i32 }] }
// CHECK-SAME: @__objc_ivar_offset_X.ivar1.i
// CHECK-SAME: @"__objc_ivar_offset_X.ivar2.\01"
// CHECK-SAME: @"__objc_ivar_offset_X.y.\01"

// Check that we get some plausible property metadata.
// CHECK: private unnamed_addr constant [5 x i8] c"Ti,R\00", align 1
// CHECK: private unnamed_addr constant [6 x i8] c"T@,Vy\00", align 1
// CHECK: = internal global { i32, i32, ptr, [2 x { ptr, ptr, ptr, ptr, ptr }] } { i32 2, i32 40, ptr null,

// Check that we get a class structure.
// CHECK: @._OBJC_CLASS_X = global { ptr, ptr, ptr, i64, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr }
// CHECK-SAME: @._OBJC_METACLASS_X
// CHECK-SAME: @._OBJC_CLASS_Super

// And check that we get a pointer to it in the right place
// CHECK: @._OBJC_REF_CLASS_X = global 
// CHECK-SAME: @._OBJC_CLASS_X
// CHECK-SAME: section "__objc_class_refs"

