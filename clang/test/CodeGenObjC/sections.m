// RUN: %clang_cc1 -triple x86_64-unknown-windows -fobjc-dispatch-method=mixed -fobjc-runtime=ios -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-COFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fobjc-dispatch-method=mixed -fobjc-runtime=ios -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-ELF
// RUN: %clang_cc1 -triple x86_64-apple-ios -fobjc-dispatch-method=mixed -fobjc-runtime=ios -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-MACHO

__attribute__((__objc_root_class__))
@interface NSObject
+ (void)load;
+ (id)class;
@end

@protocol P
@end

@interface I<P> : NSObject
+ (void)load;
@end

@implementation I
+ (void)load; { }
@end

@implementation I(C)
+ (void)load; {
  [super load];
}
@end

@interface J : NSObject
- (void)m;
@end

_Bool f(J *j) {
  [j m];
  return [I class] == @protocol(P);
}

// CHECK-COFF: @"_OBJC_$_CLASS_METHODS_I" = private
// CHECK-COFF: @"OBJC_CLASSLIST_SUP_REFS_$_" = private {{.*}}, section ".objc_superrefs$B"
// CHECK-COFF: @OBJC_SELECTOR_REFERENCES_ = private {{.*}}, section ".objc_selrefs$B"
// CHECK-COFF: @"OBJC_CLASSLIST_REFERENCES_$_" = private {{.*}}, section ".objc_classrefs$B"
// CHECK-COFF: @_objc_msgSend_fixup_class = {{.*}}, section ".objc_msgrefs$B"
// CHECK-COFF: @"_OBJC_LABEL_PROTOCOL_$_P" = {{.*}}, section ".objc_protolist$B"
// CHECK-COFF: @"_OBJC_PROTOCOL_REFERENCE_$_P" = {{.*}}, section ".objc_protorefs$B"
// CHECK-COFF: @"OBJC_LABEL_CLASS_$" = private {{.*}}, section ".objc_classlist$B"
// CHECK-COFF: @"OBJC_LABEL_NONLAZY_CLASS_$" = private {{.*}}, section ".objc_nlclslist$B"
// CHECK-COFF: @"OBJC_LABEL_CATEGORY_$" = private {{.*}}, section ".objc_catlist$B"
// CHECK-COFF: @"OBJC_LABEL_NONLAZY_CATEGORY_$" = private {{.*}}, section ".objc_nlcatlist$B"
// CHECK-COFF: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !".objc_imageinfo$B"}

// CHECK-ELF: @"_OBJC_$_CLASS_METHODS_I" = private
// CHECK-ELF: @"OBJC_CLASSLIST_SUP_REFS_$_" = private {{.*}}, section "objc_superrefs"
// CHECK-ELF: @OBJC_SELECTOR_REFERENCES_ = private {{.*}}, section "objc_selrefs"
// CHECK-ELF: @"OBJC_CLASSLIST_REFERENCES_$_" = private {{.*}}, section "objc_classrefs"
// CHECK-ELF: @_objc_msgSend_fixup_class = {{.*}}, section "objc_msgrefs"
// CHECK-ELF: @"_OBJC_LABEL_PROTOCOL_$_P" = {{.*}}, section "objc_protolist"
// CHECK-ELF: @"_OBJC_PROTOCOL_REFERENCE_$_P" = {{.*}}, section "objc_protorefs"
// CHECK-ELF: @"OBJC_LABEL_CLASS_$" = private {{.*}}, section "objc_classlist"
// CHECK-ELF: @"OBJC_LABEL_NONLAZY_CLASS_$" = private {{.*}}, section "objc_nlclslist"
// CHECK-ELF: @"OBJC_LABEL_CATEGORY_$" = private {{.*}}, section "objc_catlist"
// CHECK-ELF: @"OBJC_LABEL_NONLAZY_CATEGORY_$" = private {{.*}}, section "objc_nlcatlist"
// CHECK-ELF: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"objc_imageinfo"}

// CHECK-MACHO: @"_OBJC_$_CLASS_METHODS_I" = internal {{.*}}, section "__DATA, __objc_const"
// CHECK-MACHO: @"OBJC_CLASSLIST_SUP_REFS_$_" = private {{.*}}, section "__DATA,__objc_superrefs,regular,no_dead_strip"
// CHECK-MACHO: @OBJC_SELECTOR_REFERENCES_ = internal {{.*}}, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"
// CHECK-MACHO: @"OBJC_CLASSLIST_REFERENCES_$_" = internal {{.*}}, section "__DATA,__objc_classrefs,regular,no_dead_strip"
// CHECK-MACHO: @_objc_msgSend_fixup_class = {{.*}}, section "__DATA,__objc_msgrefs,coalesced"
// CHECK-MACHO: @"_OBJC_LABEL_PROTOCOL_$_P" = {{.*}}, section "__DATA,__objc_protolist,coalesced,no_dead_strip"
// CHECK-MACHO: @"_OBJC_PROTOCOL_REFERENCE_$_P" = {{.*}}, section "__DATA,__objc_protorefs,coalesced,no_dead_strip"
// CHECK-MACHO: @"OBJC_LABEL_CLASS_$" = private {{.*}}, section "__DATA,__objc_classlist,regular,no_dead_strip"
// CHECK-MACHO: @"OBJC_LABEL_NONLAZY_CLASS_$" = private {{.*}}, section "__DATA,__objc_nlclslist,regular,no_dead_strip"
// CHECK-MACHO: @"OBJC_LABEL_CATEGORY_$" = private {{.*}}, section "__DATA,__objc_catlist,regular,no_dead_strip"
// CHECK-MACHO: @"OBJC_LABEL_NONLAZY_CATEGORY_$" = private {{.*}}, section "__DATA,__objc_nlcatlist,regular,no_dead_strip"
// CHECK-MACHO: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}

