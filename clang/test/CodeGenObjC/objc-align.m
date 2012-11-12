// 32-bit

// RUNX: llvm-gcc -m32 -emit-llvm -S -o %t %s &&
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: grep '@"\\01L_OBJC_CATEGORY_A_Cat" = internal global .*, section "__OBJC,__category,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_CLASS_A" = internal global .*, section "__OBJC,__class,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_CLASS_C" = internal global .*, section "__OBJC,__class,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_CLASS_PROTOCOLS_C" = internal global .*, section "__OBJC,__cat_cls_meth,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_METACLASS_A" = internal global .*, section "__OBJC,__meta_class,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_METACLASS_C" = internal global .*, section "__OBJC,__meta_class,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_MODULES" = internal global .*, section "__OBJC,__module_info,regular,no_dead_strip", align 4' %t
// RUN: grep '@"\\01L_OBJC_PROTOCOL_P" = internal global .*, section "__OBJC,__protocol,regular,no_dead_strip", align 4' %t

// 64-bit

// RUNX: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o %t %s &&
// RUNX: grep '@"OBJC_CLASS_$_A" = global' %t &&
// RUNX: grep '@"OBJC_CLASS_$_C" = global' %t &&
// RUNX: grep '@"OBJC_METACLASS_$_A" = global' %t &&
// RUNX: grep '@"OBJC_METACLASS_$_C" = global' %t &&
// RUNX: grep '@"\\01L_OBJC_CLASSLIST_REFERENCES_$_0" = internal global .*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01L_OBJC_LABEL_CATEGORY_$" = internal global .*, section "__DATA, __objc_catlist, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01L_OBJC_LABEL_CLASS_$" = internal global .*, section "__DATA, __objc_classlist, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_$_CATEGORY_A_$_Cat" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_PROTOCOLS_$_C" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_RO_$_A" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_RO_$_C" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_LABEL_PROTOCOL_$_P" = weak hidden global .*, section "__DATA, __objc_protolist, coalesced, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_METACLASS_RO_$_A" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_METACLASS_RO_$_C" = internal global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_PROTOCOL_$_P" = weak hidden global .*, section "__DATA,__datacoal_nt,coalesced", align 8' %t &&


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
