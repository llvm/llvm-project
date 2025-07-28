// RUN: %clang_cc1 -triple arm64e -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

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

// CHECK: @"OBJC_EHTYPE_$_A" = global %struct._objc_typeinfo { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @objc_ehtype_vtable, i32 2), i32 2), ptr @OBJC_CLASS_NAME_, ptr @"OBJC_CLASS_$_A" }
 //.      @"OBJC_EHTYPE_$_A" = global %struct._objc_typeinfo { ptr getelementptr inbounds (ptr, ptr @objc_ehtype_vtable, i32 2), ptr @OBJC_CLASS_NAME_, ptr @"OBJC_CLASS_$_A" }
