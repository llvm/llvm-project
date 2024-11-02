// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -emit-llvm -o - %s | FileCheck -check-prefix CHECK-OSX %s
// RUN: %clang_cc1 -triple thumbv7-apple-ios3.0.0  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-IOS %s
// rdar://14802916

@interface I
@end

@implementation I @end
// CHECK-OSX: ptr null, ptr @_objc_empty_cache, ptr null
// CHECK-IOS: ptr null, ptr @_objc_empty_cache, ptr null
