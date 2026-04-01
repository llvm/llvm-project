// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -O2 -disable-llvm-passes -o - %s | FileCheck %s

#include "literal-support.h"

#if __has_feature(objc_constant_literals)

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

// Check that the constant classes were picked and emitted
// CHECK: %struct.__builtin_NSDictionary = type { ptr, i64, i64, ptr, ptr }
// CHECK: %struct.__builtin_NSConstantIntegerNumber = type { ptr, ptr, i64 }
// CHECK: %struct.__builtin_NSArray = type { ptr, i64, ptr }
// CHECK: %struct.__builtin_NSConstantDoubleNumber = type { ptr, double }
// CHECK: %struct.__builtin_NSConstantFloatNumber = type { ptr, float }

// Ensure we're able to use literals at global scope

// CHECK: @"OBJC_CLASS_$_NSConstantDictionary" = external global %struct._class_t
// CHECK: @_unnamed_array_storage = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.3 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_.2], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_ = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 1, ptr @_unnamed_array_storage, ptr @_unnamed_array_storage.3 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8
// CHECK: @dict = global ptr @_unnamed_nsdictionary_, align 8
NSDictionary *dict = @{@"fast food" : @"burger"};

// CHECK: @"OBJC_CLASS_$_NSConstantIntegerNumber" = external global %struct._class_t
// CHECK: @.str.4 = private unnamed_addr constant [2 x i8] c"i\00", align 1
// CHECK: @_unnamed_nsconstantintegernumber_ = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.4, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8
// CHECK: @__kCFBooleanTrue = external global ptr
// CHECK: @"OBJC_CLASS_$_NSConstantArray" = external global %struct._class_t
// CHECK: @_unnamed_array_storage.5 = internal unnamed_addr constant [2 x ptr] [ptr @_unnamed_nsconstantintegernumber_, ptr @__kCFBooleanTrue], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsarray_ = private constant %struct.__builtin_NSArray { ptr @"OBJC_CLASS_$_NSConstantArray", i64 2, ptr @_unnamed_array_storage.5 }, section "__DATA,__objc_arrayobj,regular,no_dead_strip", align 8
// CHECK: @arr = global ptr @_unnamed_nsarray_, align 8
NSArray *arr = @[ @42, @YES ];

// CHECK: @"OBJC_CLASS_$_NSConstantDoubleNumber" = external global %struct._class_t
// CHECK: @_unnamed_nsconstantdoublenumber_ = private constant %struct.__builtin_NSConstantDoubleNumber { ptr @"OBJC_CLASS_$_NSConstantDoubleNumber", double 6.000000e+00 }, section "__DATA,__objc_doubleobj,regular,no_dead_strip", align 8
// CHECK: @num = global ptr @_unnamed_nsconstantdoublenumber_, align 8
NSNumber *num = @(2 + 4.0);

// CHECK: @_unnamed_nsconstantintegernumber_.6 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.4, i64 17 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsconstantintegernumber_.8 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.7, i64 25 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsconstantintegernumber_.10 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.9, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsconstantintegernumber_.12 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.11, i64 97 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8

// Check globals are emitted for test_array
// CHECK: @_unnamed_array_storage.15 = internal unnamed_addr constant [2 x ptr] [ptr @_unnamed_cfstring_.14, ptr @_unnamed_nsconstantintegernumber_], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsarray_.16 = private constant %struct.__builtin_NSArray { ptr @"OBJC_CLASS_$_NSConstantArray", i64 2, ptr @_unnamed_array_storage.15 }, section "__DATA,__objc_arrayobj,regular,no_dead_strip", align 8

// Check globals are emitted for test_dictionary
// CHECK: @"OBJC_CLASS_$_NSConstantFloatNumber" = external global %struct._class_t
// CHECK: @_unnamed_nsconstantfloatnumber_ = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", float 2.200000e+01 }, section "__DATA,__objc_floatobj,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.27 = internal unnamed_addr constant [3 x ptr] [ptr @_unnamed_cfstring_.18, ptr @_unnamed_cfstring_.26, ptr @_unnamed_cfstring_.22], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.28 = internal unnamed_addr constant [3 x ptr] [ptr @_unnamed_cfstring_.20, ptr @_unnamed_nsconstantfloatnumber_, ptr @_unnamed_cfstring_.24], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.29 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 3, ptr @_unnamed_array_storage.27, ptr @_unnamed_array_storage.28 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8

// CHECK: @_unnamed_nsconstantintegernumber_.30 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.9, i64 -1 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.31 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_nsconstantintegernumber_.30], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsarray_.32 = private constant %struct.__builtin_NSArray { ptr @"OBJC_CLASS_$_NSConstantArray", i64 1, ptr @_unnamed_array_storage.31 }, section "__DATA,__objc_arrayobj,regular,no_dead_strip", align 8

// CHECK-LABEL: define void @test_numeric()
void test_numeric() {
  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsconstantintegernumber_.6)
  id ilit = @17;
  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsconstantintegernumber_.8)
  id ulit = @25u;
  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsconstantintegernumber_.10)
  id ulllit = @42ull;
  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsconstantintegernumber_.12)
  id charlit = @'a';
}

// CHECK-LABEL: define void @test_array
void test_array(id a, id b) {

  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsarray_.16)
  id arr = @[ @"meaningOfLife", @42 ];
}

// CHECK-LABEL: define void @test_dictionary
void test_dictionary(id k1, id o1, id k2, id o2) {

  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsdictionary_.29)
  id dict = @{@"fruit" : @"apple",
              @"vegetable" : @"carrot",
              @"number" : @22.0f};
}

// CHECK-LABEL: define void @test_int64array(
void test_int64array() {
  // CHECK: call ptr @llvm.objc.retain(ptr @_unnamed_nsarray_.32)
  id arr = @[@(0xFFFFFFFFFFFFFFFF)];
}

#endif
