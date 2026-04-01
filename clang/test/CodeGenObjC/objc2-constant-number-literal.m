// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK

#if __has_feature(objc_constant_literals)

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

#include <stdbool.h>
#define CTrue ((bool)1)
#define CFalse ((bool)0)

#define NAN __builtin_nanf("0x7fc00000")
#define INFINITY __builtin_huge_valf()

#include "constant-literal-support.h"

// CHECK: %struct.__builtin_NSConstantIntegerNumber = type { ptr, ptr, i64 }
// CHECK: %struct.__builtin_NSConstantFloatNumber = type { ptr, float }
// CHECK: %struct.__builtin_NSConstantDoubleNumber = type { ptr, double }

// CHECK: @_unnamed_nsconstantintegernumber_ = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str, i64 97 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0:[0-9+]]]
// CHECK: @_unnamed_nsconstantintegernumber_.2 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.1, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_nsconstantintegernumber_.3 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.1, i64 -42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_nsconstantintegernumber_.5 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.4, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_nsconstantintegernumber_.7 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.6, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_nsconstantintegernumber_.8 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.6, i64 42 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @"OBJC_CLASS_$_NSConstantFloatNumber" = external global %struct._class_t
// CHECK: @_unnamed_nsconstantfloatnumber_ = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", float 0x400921FB60000000 }, section "__DATA,__objc_floatobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @"OBJC_CLASS_$_NSConstantDoubleNumber" = external global %struct._class_t
// CHECK: @_unnamed_nsconstantdoublenumber_ = private constant %struct.__builtin_NSConstantDoubleNumber { ptr @"OBJC_CLASS_$_NSConstantDoubleNumber", double 0x400921FB54411744 }, section "__DATA,__objc_doubleobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @__kCFBooleanTrue = external global ptr #0
// CHECK: @__kCFBooleanFalse = external global ptr #0
// CHECK: @_unnamed_nsconstantintegernumber_.9 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.1, i64 1 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #0
// CHECK: @_unnamed_nsconstantintegernumber_.10 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.1, i64 0 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #0
// CHECK: @_unnamed_nsconstantfloatnumber_.11 = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", float 0x7FF8000000000000 }, section "__DATA,__objc_floatobj,regular,no_dead_strip", align 8 #0
// CHECK: @_unnamed_nsconstantfloatnumber_.12 = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", float 0x7FF0000000000000 }, section "__DATA,__objc_floatobj,regular,no_dead_strip", align 8 #0
// CHECK: @_unnamed_nsconstantfloatnumber_.13 = private constant %struct.__builtin_NSConstantFloatNumber { ptr @"OBJC_CLASS_$_NSConstantFloatNumber", float 0xFFF0000000000000 }, section "__DATA,__objc_floatobj,regular,no_dead_strip", align 8 #0
// NOTE: We expect `@((NSUInteger)2046)` to have an encoding of "Q" or `kCFNumberSInt128Type` on 64bit platforms. Since that isn't a public type `CFNumberType` will detect that and return
// CHECK: @.str.14 = private unnamed_addr constant [2 x i8] c"Q\00", align 1
// CHECK: @_unnamed_nsconstantintegernumber_.15 = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.14, i64 2049 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #0
// CHECK: @_unnamed_nsconstantdoublenumber_.16 = private constant %struct.__builtin_NSConstantDoubleNumber { ptr @"OBJC_CLASS_$_NSConstantDoubleNumber", double -0.000000e+00 }, section "__DATA,__objc_doubleobj,regular,no_dead_strip", align 8 #0

int main() {

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_, ptr %aNumber, align 8
  NSNumber *aNumber = @'a';

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_, ptr %aNumber2, align 8
  NSNumber *aNumber2 = @'a';

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.2, ptr %fortyTwo, align 8
  NSNumber *fortyTwo = @42;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.3, ptr %negativeFortyTwo, align 8
  NSNumber *negativeFortyTwo = @-42;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.2, ptr %positiveFortyTwo, align 8
  NSNumber *positiveFortyTwo = @+42;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.5, ptr %fortyTwoUnsigned, align 8
  NSNumber *fortyTwoUnsigned = @42u;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.7, ptr %fortyTwoLong, align 8
  NSNumber *fortyTwoLong = @42l;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.7, ptr %fortyTwoLong2, align 8
  NSNumber *fortyTwoLong2 = @42l;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.8, ptr %fortyTwoLongLong, align 8
  NSNumber *fortyTwoLongLong = @42ll;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.8, ptr %fortyTwoLongLong2, align 8
  NSNumber *fortyTwoLongLong2 = @42ll;

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_, ptr %piFloat, align 8
  NSNumber *piFloat = @3.141592654f;

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_, ptr %piFloat2, align 8
  NSNumber *piFloat2 = @3.141592654f;

  // CHECK: store ptr @_unnamed_nsconstantdoublenumber_, ptr %piDouble, align 8
  NSNumber *piDouble = @3.1415926535;

  // CHECK: store ptr @_unnamed_nsconstantdoublenumber_, ptr %piDouble2, align 8
  NSNumber *piDouble2 = @3.1415926535;

  // CHECK: store ptr @__kCFBooleanTrue, ptr %yesNumber, align 8
  NSNumber *yesNumber = @(YES);

  // CHECK: store ptr @__kCFBooleanFalse, ptr %noNumber, align 8
  NSNumber *noNumber = @(NO);

  // CHECK: store ptr @__kCFBooleanTrue, ptr %yesNumber1, align 8
  NSNumber *yesNumber1 = @(__objc_yes);

  // CHECK: store ptr @__kCFBooleanFalse, ptr %noNumber1, align 8
  NSNumber *noNumber1 = @(__objc_no);

  // CHECK: store ptr @__kCFBooleanTrue, ptr %c99BoolNumberTrue, align 8
  NSNumber *c99BoolNumberTrue = @CTrue;

  // CHECK: store ptr @__kCFBooleanFalse, ptr %c99BoolNumberFalse, align 8
  NSNumber *c99BoolNumberFalse = @CFalse;

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.9, ptr %trueLikeBoolNumber, align 8
  NSNumber *trueLikeBoolNumber = @(1);

  // CHECK: store ptr @_unnamed_nsconstantintegernumber_.10, ptr %falseLikeBoolNumber, align 8
  NSNumber *falseLikeBoolNumber = @(0);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.11, ptr %nanNumber, align 8
  NSNumber *nanNumber = @(NAN);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.11, ptr %nanNumber2, align 8
  NSNumber *nanNumber2 = @(NAN);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.12, ptr %infNumber, align 8
  NSNumber *infNumber = @(INFINITY);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.12, ptr %infNumber2, align 8
  NSNumber *infNumber2 = @(INFINITY);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.13, ptr %negInfNumber, align 8
  NSNumber *negInfNumber = @(-INFINITY);

  // CHECK: store ptr @_unnamed_nsconstantfloatnumber_.13, ptr %negInfNumber2, align 8
  NSNumber *negInfNumber2 = @(-INFINITY);

  // CHECK: @_unnamed_nsconstantintegernumber_.15, ptr %unsginedLongLongQEncoded, align 8
  NSNumber *unsginedLongLongQEncoded = @((NSUInteger)2049); // NOTE: On 64bit platforms we expect this to have a "Q" encoding, on 32bit we expect this to have a "q" encoding.

  // CHECK: store ptr @_unnamed_nsconstantdoublenumber_.16, ptr %negZero, align 8
  NSNumber *negZero = @(-0.0);

  // CHECK: store ptr @_unnamed_nsconstantdoublenumber_.16, ptr %negZero2, align 8
  NSNumber *negZero2 = @(-0.0);

  return 0;
}

// CHECK: attributes #[[ATTR0]] = { "objc_arc_inert" }

#endif
