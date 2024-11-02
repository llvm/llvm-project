// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s


@class NSString;

@interface ANObject {
@public
// Public ivars have default visibility
// CHECK: @"__objc_ivar_offset_ANObject.isa.\01" = global i32 0
  struct objc_object *isa;
@private
// Private and package ivars should have hidden linkage.
// Check that in the GNUstep v2 ABI, instance variable offset names include
// type encodings (with @ mangled to \01 to avoid collisions with ELF symbol
// versions).
// CHECK: private unnamed_addr constant [12 x i8] c"@\22NSString\22\00"
// CHECK: @"__objc_ivar_offset_ANObject._stringIvar.\01" = hidden global i32 8
  NSString    *_stringIvar;
@package
// CHECK: @__objc_ivar_offset_ANObject._intIvar.i = hidden global i32 16
  int         _intIvar;
  _Bool       boolIvar;
}
@end
@implementation ANObject @end

// Check that the ivar metadata contains 3 entries of the correct form and correctly sets the size.
// CHECK: @.objc_ivar_list = private global { i32, i64, [4 x { ptr, ptr, ptr, i32, i32 }] } { i32 4, i64 32,
// Check that we emit 1 as the size of _Bool, not 0.
// CHECK-SAME:  @__objc_ivar_offset_ANObject.boolIvar.B, i32 1, i32 4
