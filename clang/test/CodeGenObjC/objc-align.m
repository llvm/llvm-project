// 32-bit

// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// CHECK: @OBJC_METACLASS_A = private global {{.*}}, section "__OBJC,__meta_class,regular,no_dead_strip", align 4
// CHECK: @OBJC_CLASS_A = private global {{.*}}, section "__OBJC,__class,regular,no_dead_strip", align 4
// CHECK: @OBJC_CATEGORY_A_Cat = private global {{.*}}, section "__OBJC,__category,regular,no_dead_strip", align 4
// CHECK: @OBJC_PROTOCOL_P = private global {{.*}}, section "__OBJC,__protocol,regular,no_dead_strip", align 4
// CHECK: @OBJC_CLASS_PROTOCOLS_C = private global {{.*}}, section "__OBJC,__cat_cls_meth,regular,no_dead_strip", align 4
// CHECK: @OBJC_METACLASS_C = private global {{.*}}, section "__OBJC,__meta_class,regular,no_dead_strip", align 4
// CHECK: @OBJC_CLASS_C = private global {{.*}}, section "__OBJC,__class,regular,no_dead_strip", align 4
// CHECK: @OBJC_MODULES = private global {{.*}}, section "__OBJC,__module_info,regular,no_dead_strip", align 4


@interface A @end
@implementation A
@end
@implementation A (Cat)
@end
@protocol P
@end
@interface C <P>
@end
@implementation C
@end
