// RUN: %clang_cc1 -triple arm64e-apple-macosx26.0.0 -fobjc-runtime=macosx-26.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -fptrauth-intrinsics -fptrauth-calls -fptrauth-objc-isa -I %S/Inputs -emit-llvm -o - %s | FileCheck %s
// rdar://174359070

#include "constant-literal-support.h"

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

// Check that isa pointers in all ObjC constant literal structs are signed with
// ptrauth (key 2, discriminator 0x6AE1 = 27361, address-discriminated).

// CHECK: @"OBJC_CLASS_$_NSConstantIntegerNumber.ptrauth" = private constant { ptr, i32, i64, i64 } { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", i32 2, i64 ptrtoint (ptr @_unnamed_nsconstantintegernumber_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_nsconstantintegernumber_ = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber.ptrauth", ptr @.str, i64 42 }
// CHECK: @"OBJC_CLASS_$_NSConstantFloatNumber.ptrauth" = private constant { ptr, i32, i64, i64 } { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", i32 2, i64 ptrtoint (ptr @_unnamed_nsconstantfloatnumber_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_nsconstantfloatnumber_ = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber.ptrauth", float 0x40091EB860000000 }
// CHECK: @"OBJC_CLASS_$_NSConstantDoubleNumber.ptrauth" = private constant { ptr, i32, i64, i64 } { ptr @"OBJC_CLASS_$_NSConstantDoubleNumber", i32 2, i64 ptrtoint (ptr @_unnamed_nsconstantdoublenumber_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_nsconstantdoublenumber_ = private constant %struct.__builtin_NSConstantDoubleNumber { ptr @"OBJC_CLASS_$_NSConstantDoubleNumber.ptrauth", double 3.140000e+00 }
// CHECK: @__CFConstantStringClassReference.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @__CFConstantStringClassReference, i32 2, i64 ptrtoint (ptr @_unnamed_cfstring_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference.ptrauth, i32 1992, ptr @.str.1, i64 3 }
// CHECK: @"OBJC_CLASS_$_NSConstantArray.ptrauth" = private constant { ptr, i32, i64, i64 } { ptr @"OBJC_CLASS_$_NSConstantArray", i32 2, i64 ptrtoint (ptr @_unnamed_nsarray_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_nsarray_ = private constant %struct.__builtin_NSArray { ptr @"OBJC_CLASS_$_NSConstantArray.ptrauth", i64 1, ptr @_unnamed_array_storage }
// CHECK: @"OBJC_CLASS_$_NSConstantDictionary.ptrauth" = private constant { ptr, i32, i64, i64 } { ptr @"OBJC_CLASS_$_NSConstantDictionary", i32 2, i64 ptrtoint (ptr @_unnamed_nsdictionary_ to i64), i64 27361 }, section "llvm.ptrauth"
// CHECK: @_unnamed_nsdictionary_ = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary.ptrauth", i64 1, i64 2, ptr @_unnamed_array_storage.12, ptr @_unnamed_array_storage.13 }

int main() {
  NSNumber *n = @42;
  NSNumber *f = @3.14f;
  NSNumber *d = @3.14;
  NSNumber *b = @YES;
  NSArray *a = @[ @"foo" ];
  NSDictionary *dict = @{ @"a" : @1, @"b" : @2 };
  return 0;
}
