// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s --check-prefix=CHECK



#if __has_feature(objc_constant_literals)

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

#include "constant-literal-support.h"

// CHECK: %struct.__builtin_NSDictionary = type { ptr, i64, i64, ptr, ptr }
// CHECK: %struct.__builtin_NSArray = type { ptr, i64, ptr }

/// This is only for reference if the sorting changes... (see the dictionary checks below)
// CHECK: @.str = private unnamed_addr constant [6 x i8] c"zebra\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str, i64 5 }, section "__DATA,__cfstring", align 8 #[[ATTR0:[0-9+]]]
// CHECK: @.str.2 = private unnamed_addr constant [6 x i8] c"horse\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @_unnamed_cfstring_.3 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.2, i64 5 }, section "__DATA,__cfstring", align 8 #[[ATTR0]]
// CHECK: @.str.5 = private unnamed_addr constant [5 x i8] c"bear\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @_unnamed_cfstring_.6 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.5, i64 4 }, section "__DATA,__cfstring", align 8 #[[ATTR0]]
// CHECK: @.str.8 = private unnamed_addr constant [6 x i8] c"apple\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @_unnamed_cfstring_.9 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.8, i64 5 }, section "__DATA,__cfstring", align 8 #[[ATTR0]]
// CHECK: @"OBJC_CLASS_$_NSConstantDictionary" = external global %struct._class_t

/// This checks that we're sorting things as expected with one not sorted (first check) and the other pre-sorted (second check)
// CHECK: @_unnamed_array_storage = internal unnamed_addr constant [4 x ptr] [ptr @_unnamed_cfstring_.9, ptr @_unnamed_cfstring_.6, ptr @_unnamed_cfstring_.3, ptr @_unnamed_cfstring_], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_ = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 4, ptr @_unnamed_array_storage, ptr @_unnamed_array_storage.11 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_array_storage.12 = internal unnamed_addr constant [4 x ptr] [ptr @_unnamed_cfstring_.9, ptr @_unnamed_cfstring_.6, ptr @_unnamed_cfstring_.3, ptr @_unnamed_cfstring_], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.14 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 4, ptr @_unnamed_array_storage.12, ptr @_unnamed_array_storage.13 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]

// CHECK: @_unnamed_array_storage.18 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_.16], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.19 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_nsconstantintegernumber_.17], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.20 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 1, ptr @_unnamed_array_storage.18, ptr @_unnamed_array_storage.19 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_array_storage.21 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_.16], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.22 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_nsconstantintegernumber_.17], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.23 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 1, ptr @_unnamed_array_storage.21, ptr @_unnamed_array_storage.22 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_array_storage.24 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_.16], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.25 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_nsconstantintegernumber_.17], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.26 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 1, ptr @_unnamed_array_storage.24, ptr @_unnamed_array_storage.25 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @_unnamed_array_storage.27 = internal unnamed_addr constant [1 x ptr] [ptr @_unnamed_cfstring_.16], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_array_storage.28 = internal unnamed_addr constant [1 x ptr] [ptr @__kCFBooleanTrue], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsdictionary_.29 = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 1, ptr @_unnamed_array_storage.27, ptr @_unnamed_array_storage.28 }, section "__DATA,__objc_dictobj,regular,no_dead_strip", align 8 #[[ATTR0]]

// CHECK: @"OBJC_CLASS_$_NSConstantArray" = external global %struct._class_t
// CHECK: @_unnamed_array_storage.36 = internal unnamed_addr constant [3 x ptr] [ptr @_unnamed_cfstring_.31, ptr @_unnamed_cfstring_.33, ptr @_unnamed_cfstring_.35], section "__DATA,__objc_arraydata,regular,no_dead_strip", align 8
// CHECK: @_unnamed_nsarray_ = private constant %struct.__builtin_NSArray { ptr @"OBJC_CLASS_$_NSConstantArray", i64 3, ptr @_unnamed_array_storage.36 }, section "__DATA,__objc_arrayobj,regular,no_dead_strip", align 8 #[[ATTR0]]
// CHECK: @__NSArray0__struct = external global ptr #[[ATTR0]]
// CHECK: @__NSDictionary0__struct = external global ptr #[[ATTR0]]

// Ensure we're still promoting constant literals even when the collection is downgraded, e.g `@{@NO : @"no"}`
// CHECK: @__kCFBooleanFalse = external global ptr #[[ATTR0]]

// Ensure we're falling back on the non-plist like dictionary case
// CHECK: @"OBJC_CLASS_$_NSDictionary" = external global %struct._class_t

/// Dictionary and Array literals now allowed at global scope
static NSDictionary *const wootwootDict = @{@"name" : @YES, @"apple" : @NO};
static NSArray *const hollahollaArray = @[ @"At", @"Your", @"Boy" ];

/// Empty Dictionary and Array literals map to global constants
static NSArray *const emptyArray = @[];
static NSDictionary *const emptyDictionary = @{};

static NSString *const someStringConstantVar = @"foo";
static NSNumber *const someNumberConstantVar = @2;

int main() {

  /// Make sure that we're sorting before emitting is sorting string keys as expected.
  // CHECK: store ptr @_unnamed_nsdictionary_, ptr %alphaSortTestDict, align 8
  NSDictionary *alphaSortTestDict = @{@"zebra" : @26,
                                      @"horse" : @8,
                                      @"bear" : @2,
                                      @"apple" : @1};
  // CHECK: store ptr @_unnamed_nsdictionary_.14, ptr %alphaSortTestDict2, align 8
  NSDictionary *alphaSortTestDict2 = @{@"bear" : @2,
                                       @"apple" : @1,
                                       @"horse" : @8,
                                       @"zebra" : @26};

  // CHECK: store ptr @_unnamed_nsdictionary_.20, ptr %dict, align 8
  NSDictionary *const dict = @{@"name" : @666};
  // CHECK: store ptr @_unnamed_nsdictionary_.23, ptr %dict1, align 8
  NSDictionary *dict1 = @{@"name" : @666};
  // CHECK: store ptr @_unnamed_nsdictionary_.26, ptr %dict2, align 8
  NSDictionary *const dict2 = @{@"name" : @666};

  // CHECK: store ptr @_unnamed_nsdictionary_.29, ptr %wootwootDict1, align 8
  NSDictionary *const wootwootDict1 = @{@"name" : @YES};
  // CHECK: store ptr @_unnamed_nsarray_, ptr %hollahollaArray1, align 8
  NSArray *const hollahollaArray1 = @[ @"At", @"Your", @"Boy" ];

  // CHECK: store ptr @__NSArray0__struct, ptr %emptyArray1, align 8
  NSArray *emptyArray1 = @[];
  // CHECK: store ptr @__NSDictionary0__struct, ptr %emptyDictionary1, align 8
  NSDictionary *emptyDictionary1 = @{};

  /// Non String dictionaries should be downgraded to normal types that have an objc_msgSend
  // CHECK: @objc_msgSend
  NSDictionary *const nonPlistTypeDict = @{@NO : @"no"};

  /// For now we only support raw literals in collections not references to other varibles that *could* be modified
  /// so ensure this get's downgraded to a normal runtime literal

  // CHECK: @objc_msgSend
  NSArray *const nonConstDueToVarReferenceArray = @[ @"baz", someStringConstantVar, someNumberConstantVar ];

  // CHECK: @objc_msgSend
  NSDictionary *const nonConstDueToVarReferenceDictionary = @{@"baz" : someStringConstantVar, @"blah" : someNumberConstantVar};

  return 0;
}

// CHECK: attributes #[[ATTR0]] = { "objc_arc_inert" }

#endif
