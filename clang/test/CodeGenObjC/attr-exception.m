// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -fobjc-exceptions -fvisibility hidden -o - %s | FileCheck -check-prefix=CHECK-HIDDEN %s

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
// CHECK: @"OBJC_EHTYPE_$_A" ={{.*}} global {{%.*}} { i8** getelementptr inbounds (i8*, i8** @objc_ehtype_vtable, i32 2)
// CHECK-HIDDEN: @"OBJC_EHTYPE_$_A" = hidden global {{%.*}} { i8** getelementptr inbounds (i8*, i8** @objc_ehtype_vtable, i32 2)

__attribute__((objc_exception))
__attribute__((visibility("default")))
@interface B : Root
@end

@implementation B
@end
// CHECK: @"OBJC_EHTYPE_$_B" ={{.*}} global {{%.*}} { i8** getelementptr inbounds (i8*, i8** @objc_ehtype_vtable, i32 2)
// CHECK-HIDDEN: @"OBJC_EHTYPE_$_B" ={{.*}} global {{%.*}} { i8** getelementptr inbounds (i8*, i8** @objc_ehtype_vtable, i32 2)
