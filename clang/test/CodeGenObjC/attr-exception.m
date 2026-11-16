// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -fobjc-exceptions -fvisibility=hidden -o - %s | FileCheck -check-prefix=CHECK-HIDDEN %s

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
// CHECK: @"OBJC_EHTYPE_$_A" ={{.*}} global {{%.*}} { ptr getelementptr inbounds (i8, ptr @objc_ehtype_vtable, i64 16)
// CHECK-HIDDEN: @"OBJC_EHTYPE_$_A" = hidden global {{%.*}} { ptr getelementptr inbounds (i8, ptr @objc_ehtype_vtable, i64 16)

__attribute__((objc_exception))
__attribute__((visibility("default")))
@interface B : Root
@end

@implementation B
@end
// CHECK: @"OBJC_EHTYPE_$_B" ={{.*}} global {{%.*}} { ptr getelementptr inbounds (i8, ptr @objc_ehtype_vtable, i64 16)
// CHECK-HIDDEN: @"OBJC_EHTYPE_$_B" ={{.*}} global {{%.*}} { ptr getelementptr inbounds (i8, ptr @objc_ehtype_vtable, i64 16)
