// RUN: %clang_cc1 -fvisibility hidden "-triple" "x86_64-apple-darwin8.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_4 %s
// RUN: %clang_cc1 -fvisibility hidden "-triple" "x86_64-apple-darwin9.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_5 %s
// RUN: %clang_cc1 -fvisibility hidden "-triple" "x86_64-apple-darwin10.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_6 %s

// CHECK-10_4: @"OBJC_CLASS_$_WeakClass1" = extern_weak global
// CHECK-10_5: @"OBJC_CLASS_$_WeakClass1" = external global
// CHECK-10_6: @"OBJC_CLASS_$_WeakClass1" = external global
__attribute__((availability(macosx,introduced=10.5)))
@interface WeakClass1 @end

@implementation WeakClass1(MyCategory) @end

@implementation WeakClass1(YourCategory) @end

// CHECK-10_4: @"OBJC_CLASS_$_WeakClass2" = extern_weak global 
// CHECK-10_5: @"OBJC_CLASS_$_WeakClass2" = extern_weak global 
// CHECK-10_6: @"OBJC_CLASS_$_WeakClass2" = external global 
__attribute__((availability(macosx,introduced=10.6)))
@interface WeakClass2 @end

@implementation WeakClass2(MyCategory) @end

@implementation WeakClass2(YourCategory) @end

// CHECK-10_4: @"OBJC_CLASS_$_WeakClass3" = extern_weak global
// CHECK-10_5: @"OBJC_CLASS_$_WeakClass3" = extern_weak global
// CHECK-10_6: @"OBJC_CLASS_$_WeakClass3" = external global
__attribute__((availability(macosx,introduced=10.6)))
@interface WeakClass3 @end
@class WeakClass3;

@implementation WeakClass3(MyCategory) @end

@implementation WeakClass3(YourCategory) @end

