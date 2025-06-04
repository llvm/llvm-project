// RUN: %clang_cc1 -triple arm64-apple-ios                                                   -fptrauth-calls -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-calls -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck --check-prefix=TYPEDISC_TYPEINFO %s

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

            // CHECK: @"OBJC_EHTYPE_$_A" = global {{%.*}} { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @objc_ehtype_vtable, i32 2), i32 2),
// TYPEDISC_TYPEINFO: @"OBJC_EHTYPE_$_A" = global {{%.*}} { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @objc_ehtype_vtable, i32 2), i32 2, i64 45546, ptr @"OBJC_EHTYPE_$_A"),
