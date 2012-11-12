// RUNX: llvm-gcc -m64 -emit-llvm -S -o %t %s &&
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
// RUN: grep '@".01L_OBJC_LABEL_NONLAZY_CLASS_$" = internal global \[1 x .*\] .*@"OBJC_CLASS_$_A".*, section "__DATA, __objc_nlclslist, regular, no_dead_strip", align 8' %t
// RUN: grep '@".01L_OBJC_LABEL_NONLAZY_CATEGORY_$" = internal global \[1 x .*\] .*@".01l_OBJC_$_CATEGORY_A_$_Cat".*, section "__DATA, __objc_nlcatlist, regular, no_dead_strip", align 8' %t

@interface A @end
@implementation A
+(void) load {
}
@end

@interface A (Cat) @end
@implementation A (Cat)
+(void) load {
}
@end

@interface B @end
@implementation B
-(void) load {
}
@end

@interface B (Cat) @end
@implementation B (Cat)
-(void) load {
}
@end

@interface C : A @end
@implementation C
@end
