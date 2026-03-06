// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -fobjc-runtime=macosx-10.14.0 -I %S/Inputs -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefix CHECK-ALL-DISABLED %s

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -I %S/Inputs -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefix CHECK-ALL-DISABLED-CONST-ON %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -I %S/Inputs -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefix CHECK-NUMBERS-DISABLED %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -I %S/Inputs -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefix CHECK-DICT-DISABLED %s

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

#include "constant-literal-support.h"

// CHECK-ALL-DISABLED: @__NSArray0__ = external global ptr
// CHECK-ALL-DISABLED: @__NSDictionary0__ = external global ptr
// CHECK-ALL-DISABLED-CONST-ON: @__kCFBooleanTrue = external global ptr
// CHECK-ALL-DISABLED-CONST-ON: @__NSArray0__struct = external global ptr
// CHECK-ALL-DISABLED-CONST-ON: @__NSDictionary0__struct = external global ptr
// CHECK-NUMBERS-DISABLED: @__kCFBooleanTrue = external global ptr
// CHECK-NUMBERS-DISABLED: @__NSArray0__struct = external global ptr
// CHECK-NUMBERS-DISABLED: @__NSDictionary0__struct = external global ptr
// CHECK-DICT-DISABLED: __kCFBooleanTrue = external global ptr
// CHECK-DICT-DISABLED: __NSArray0__struct = external global ptr
// CHECK-DICT-DISABLED: __NSDictionary0__struct = external global ptr

int main() {

  // CHECK-ALL-DISABLED: %[[V0:.*]] = getelementptr inbounds [1 x ptr], ptr %keys, i64 0, i64 0
  // CHECK-ALL-DISABLED: store ptr @_unnamed_cfstring_, ptr %[[V0]], align 8
  // CHECK-ALL-DISABLED: getelementptr inbounds [1 x ptr], ptr %objects, i64 0, i64 0
  // CHECK-ALL-DISABLED: load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V0:.*]] = getelementptr inbounds [1 x ptr], ptr %keys, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_, ptr %[[V0]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V1:.*]] = getelementptr inbounds [1 x ptr], ptr %objects, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @__kCFBooleanTrue, ptr %[[V1]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V2:.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V3:.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[CALL:.*]] = call ptr @objc_msgSend(ptr %[[V2]], ptr %[[V3]], ptr %objects, ptr %keys, i64 1)
  // CHECK-ALL-DISABLED-CONST-ON: store ptr %[[CALL]], ptr %wootwootDict1, align 8
  // CHECK-DICT-DISABLED: __kCFBooleanTrue
  NSDictionary *const wootwootDict1 = @{@"name" : @(YES)};


  // CHECK-ALL-DISABLED-CONST-ON: %[[V4:.*]] = getelementptr inbounds [1 x ptr], ptr %keys2, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_, ptr %[[V4]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V5:.*]] = getelementptr inbounds [1 x ptr], ptr %objects1, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: %[[V6:.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.1", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V7:.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.3, align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[CALL3:.*]] = call ptr @objc_msgSend(ptr %[[V6]], ptr %[[V7]], i32 1001)
  // CHECK-ALL-DISABLED-CONST-ON: store ptr %[[CALL3]], ptr %[[V5]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V8:.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V9:.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[CALL4:.*]] = call ptr @objc_msgSend(ptr %[[V8]], ptr %[[V9]], ptr %objects1, ptr %keys2, i64 1)
  // CHECK-ALL-DISABLED-CONST-ON: store ptr %[[CALL4]], ptr %wootwootDict2, align 8
  // CHECK-DICT-DISABLED: @_unnamed_nsconstantintegernumber_
  // CHECK-NUMBERS-DISABLED: @objc_msgSend
  NSDictionary *wootwootDict2 = @{@"name" : @(1001)};

  // CHECK-ALL-DISABLED-CONST-ON: %[[V10:.*]] = getelementptr inbounds [3 x ptr], ptr %objects5, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_.5, ptr %[[V10]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V11:.*]] = getelementptr inbounds [3 x ptr], ptr %objects5, i64 0, i64 1
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_.7, ptr %[[V11]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V12:.*]] = getelementptr inbounds [3 x ptr], ptr %objects5, i64 0, i64 2
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_.9, ptr %[[V12]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V13:.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.10", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V14:.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.12, align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[CALL6:.*]] = call ptr @objc_msgSend(ptr %[[V13]], ptr %[[V14]], ptr %objects5, i64 3)
  // CHECK-ALL-DISABLED-CONST-ON: store ptr %[[CALL6]], ptr %hollahollaArray1, align 8
  // CHECK-NUMBERS-DISABLED: store ptr @_unnamed_nsarray_, ptr %hollahollaArray1, align 8
  NSArray *const hollahollaArray1 = @[ @"At", @"Your", @"Boy" ];

  // CHECK-DICT-DISABLED: store ptr @_unnamed_nsarray_.9, ptr %hollahollaArray2, align 8
  // CHECK-NUMBERS-DISABLED: store ptr @_unnamed_nsarray_.13, ptr %hollahollaArray2, align 8
  NSArray *hollahollaArray2 = @[ @"At", @"Your", @"Boy" ];

  // We still expect the empty collection singletons to be used so long as the target supports them but not if disabled
  // CHECK-ALL-DISABLED: load ptr, ptr @__NSArray0__, align 8
  // CHECK-ALL-DISABLED: store ptr %{{.*}}, ptr %emptyArray1, align 8
  // CHECK-ALL-DISABLED: load ptr, ptr @__NSDictionary0__, align 8
  // CHECK-ALL-DISABLED: store ptr %{{.*}}, ptr %emptyDictionary1, align 8
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @__NSArray0__struct, ptr %emptyArray1, align 8
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @__NSDictionary0__struct, ptr %emptyDictionary1, align 8
  NSArray *emptyArray1 = @[];
  NSDictionary *emptyDictionary1 = @{};

  // CHECK-ALL-DISABLED-CONST-ON: %[[V20:.*]] = getelementptr inbounds [1 x ptr], ptr %keys10, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @__kCFBooleanTrue, ptr %[[V20]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V21:.*]] = getelementptr inbounds [1 x ptr], ptr %objects9, i64 0, i64 0
  // CHECK-ALL-DISABLED-CONST-ON: store ptr @_unnamed_cfstring_.14, ptr %[[V21]], align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V22:.*]] = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[V23:.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  // CHECK-ALL-DISABLED-CONST-ON: %[[CALL11:.*]] = call ptr @objc_msgSend(ptr %[[V22]], ptr %[[V23]], ptr %objects9, ptr %keys10, i64 1)
  // CHECK-ALL-DISABLED-CONST-ON: store ptr %[[CALL11]], ptr %nonPlistTypeDict, align 8
  // CHECK-DICT-DISABLED: __kCFBooleanTrue
  NSDictionary *const nonPlistTypeDict = @{@YES : @"no"};

  return 0;
}
