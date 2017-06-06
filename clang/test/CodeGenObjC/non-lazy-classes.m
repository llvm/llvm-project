// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -emit-llvm -o - %s | \
// RUN: FileCheck %s
// CHECK: @"OBJC_LABEL_NONLAZY_CLASS_$" = private global [1 x {{.*}}] {{.*}}@"OBJC_CLASS_$_A"{{.*}}, section "__DATA,__objc_nlclslist,regular,no_dead_strip", align 8
// CHECK: @"OBJC_LABEL_NONLAZY_CATEGORY_$" = private global [1 x {{.*}}] {{.*}}@"\01l_OBJC_$_CATEGORY_A_$_Cat"{{.*}}, section "__DATA,__objc_nlcatlist,regular,no_dead_strip", align 8

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
