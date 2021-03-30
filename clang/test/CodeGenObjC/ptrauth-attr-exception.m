// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

__attribute__((objc_root_class))
@interface Root {
  Class isa;
}
@end

__attribute__((objc_exception))
@interface A : Root
@end

@implementation A
@end

// CHECK: @objc_ehtype_vtable.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @objc_ehtype_vtable, i32 2) to i8*), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @"OBJC_EHTYPE_$_A" = global {{%.*}} { i8** getelementptr inbounds ({ i8*, i32, i64, i64 }, { i8*, i32, i64, i64 }* @objc_ehtype_vtable.ptrauth, i32 0, i32 0)
