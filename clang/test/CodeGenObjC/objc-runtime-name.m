// RUN: %clang_cc1 -triple i386-apple-macosx10.13.0 -fobjc-runtime=macosx-fragile-10.13.0 -fobjc-subscripting-legacy-runtime -emit-llvm -o - %s | FileCheck %s

// Check that the runtime name is emitted and used instead of the class
// identifier.

// CHECK: module asm {{.*}}objc_class_name_XYZ=0
// CHECK: module asm {{.*}}globl .objc_class_name_XYZ
// CHECK: module asm {{.*}}lazy_reference .objc_class_name_XYZ

// CHECK: @[[OBJC_CLASS_NAME:.*]] = private unnamed_addr constant [4 x i8] c"XYZ{{.*}}, section "__TEXT,__cstring,cstring_literals",
// CHECK: = private global {{.*}} @[[OBJC_CLASS_NAME]], section "__OBJC,__cls_refs,literal_pointers,no_dead_strip"

__attribute__((objc_root_class,objc_runtime_name("XYZ")))
@interface A
+(void)m1;
@end

@implementation A
+(void)m1 {}
@end

void test(void) {
  [A m1];
}
