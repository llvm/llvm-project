// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed -emit-llvm -o %t %s
// RUNX: llvm-gcc -m64 -emit-llvm -S -o %t %s &&

// RUN: grep '@"OBJC_CLASS_$_A" = global' %t
// RUN: grep '@"OBJC_CLASS_$_B" = external global' %t
// RUN: grep '@"OBJC_IVAR_$_A._ivar" = global .* section "__DATA, __objc_ivar", align 8' %t
// RUN: grep '@"OBJC_METACLASS_$_A" = global .* section "__DATA, __objc_data", align 8' %t
// RUN: grep '@"\\01L_OBJC_CLASSLIST_REFERENCES_$_[0-9]*" = internal global .* section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8' %t
// RUN: grep '@"\\01L_OBJC_CLASSLIST_SUP_REFS_$_[0-9]*" = internal global .* section "__DATA, __objc_superrefs, regular, no_dead_strip", align 8' %t | count 2
// RUN: grep '@"\\01L_OBJC_CLASS_NAME_[0-9]*" = internal global .* section "__TEXT,__objc_classname,cstring_literals", align 1' %t
// RUN: grep '@"\\01L_OBJC_LABEL_CATEGORY_$" = internal global .* section "__DATA, __objc_catlist, regular, no_dead_strip", align 8' %t
// RUN: grep '@"\\01L_OBJC_LABEL_CLASS_$" = internal global .* section "__DATA, __objc_classlist, regular, no_dead_strip", align 8' %t
// RUN: grep '@"\\01L_OBJC_METH_VAR_NAME_[0-9]*" = internal global .* section "__TEXT,__objc_methname,cstring_literals", align 1' %t
// RUN: grep '@"\\01L_OBJC_METH_VAR_TYPE_[0-9]*" = internal global .* section "__TEXT,__objc_methtype,cstring_literals", align 1' %t
// RUN: grep '@"\\01L_OBJC_PROP_NAME_ATTR_[0-9]*" = internal global .* section "__TEXT,__cstring,cstring_literals", align 1' %t
// RUN: grep '@"\\01L_OBJC_SELECTOR_REFERENCES_*" = internal global .* section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"' %t
// RUN: grep '@"\\01l_OBJC_$_CATEGORY_A_$_Cat" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_CATEGORY_CLASS_METHODS_A_$_Cat" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_CATEGORY_INSTANCE_METHODS_A_$_Cat" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_CLASS_METHODS_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_INSTANCE_METHODS_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_INSTANCE_VARIABLES_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_PROP_LIST_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_PROTOCOL_CLASS_METHODS_P" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_$_PROTOCOL_INSTANCE_METHODS_P" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_CLASS_PROTOCOLS_$_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_CLASS_RO_$_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_LABEL_PROTOCOL_$_P" = weak hidden global .* section "__DATA, __objc_protolist, coalesced, no_dead_strip", align 8' %t
// RUN: grep '@"\\01l_OBJC_METACLASS_RO_$_A" = internal global .* section "__DATA, __objc_const", align 8' %t
// RUN: grep '@"\\01l_OBJC_PROTOCOL_$_P" = weak hidden global .* section "__DATA,__datacoal_nt,coalesced", align 8' %t
// RUN: grep '@"\\01l_objc_msgSend_fixup_alloc" = weak hidden global .* section "__DATA, __objc_msgrefs, coalesced", align 16' %t
// RUN: grep '@_objc_empty_cache = external global' %t
// RUN: grep '@_objc_empty_vtable = external global' %t
// RUN: grep '@objc_msgSend_fixup(' %t
// RUN: grep '@objc_msgSend_fpret(' %t


/*

Here is a handy command for looking at llvm-gcc's output:
llvm-gcc -m64 -emit-llvm -S -o - metadata-symbols-64.m | \
  grep '=.*global' | \
  sed -e 's#global.*, section#global ... section#' | \
  sort

*/

@interface B
@end
@interface C
@end

@protocol P
+(void) fm0;
-(void) im0;
@end

@interface A<P> {
  int _ivar;
}
 
@property (assign) int ivar;

+(void) fm0;
-(void) im0;
@end

@implementation A
@synthesize ivar = _ivar;
+(void) fm0 {
}
-(void) im0 {
}
@end

@implementation A (Cat)
+(void) fm1 {
}
-(void) im1 {
}
@end

@interface D : A
@end

@implementation D
+(void) fm2 {
  [super fm1];
}
-(void) im2 {
  [super im1];
}
@end

// Test for FP dispatch method APIs
@interface Example 
@end

float FLOAT;
double DOUBLE;
long double LONGDOUBLE;
id    ID;

@implementation Example
 - (double) RET_DOUBLE
   {
        return DOUBLE;
   }
 - (float) RET_FLOAT
   {
        return FLOAT;
   }
 - (long double) RET_LONGDOUBLE
   {
        return LONGDOUBLE;
   }
@end

void *f0(id x) {
   Example* pe;
   double dd = [pe RET_DOUBLE];
   dd = [pe RET_FLOAT];
   dd = [pe RET_LONGDOUBLE];

   [B im0];
   [C im1];
   [D alloc];
}

