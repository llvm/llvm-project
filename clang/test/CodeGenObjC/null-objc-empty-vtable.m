// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.9 -emit-llvm -o - %s | FileCheck -check-prefix CHECK-OSX %s
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv7-apple-ios3.0.0  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-IOS %s
// rdar://14802916

@interface I
@end

@implementation I @end
// CHECK-OSX: %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null
// CHECK-IOS: %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null
