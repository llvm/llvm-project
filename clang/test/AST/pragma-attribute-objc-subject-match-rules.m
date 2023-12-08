// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump "-DSUBJECT=objc_interface" %s | FileCheck --check-prefix=CHECK-OBJC_INTERFACE %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=objc_protocol" %s | FileCheck --check-prefix=CHECK-OBJC_PROTOCOL %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump "-DSUBJECT=objc_category" %s | FileCheck --check-prefix=CHECK-OBJC_CATEGORY %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=objc_method" %s | FileCheck --check-prefix=CHECK-OBJC_METHOD %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=objc_method(is_instance)" %s | FileCheck --check-prefix=CHECK-OBJC_METHOD_IS_INSTANCE %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=field" %s | FileCheck --check-prefix=CHECK-FIELD %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=objc_property" %s | FileCheck --check-prefix=CHECK-OBJC_PROPERTY %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=block" %s | FileCheck --check-prefix=CHECK-BLOCK %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -Wno-objc-root-class -fsyntax-only -ast-dump -ast-dump-filter test "-DSUBJECT=hasType(functionType)" %s | FileCheck --check-prefix=CHECK-HAS_TYPE_FUNCTION_TYPE %s

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = any(SUBJECT))

@interface testInterface
@end
// CHECK-OBJC_INTERFACE: ObjCInterfaceDecl{{.*}} testInterface
// CHECK-OBJC_INTERFACE-NEXT: AnnotateAttr{{.*}} "test"

@interface testInterface ()
@end
// CHECK-OBJC_INTERFACE: ObjCCategoryDecl
// CHECK-OBJC_INTERFACE-NOT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_CATEGORY: ObjCCategoryDecl
// CHECK-OBJC_CATEGORY-NEXT: ObjCInterface
// CHECK-OBJC_CATEGORY-NEXT: AnnotateAttr{{.*}} "test"

@interface testInterface (testCategory)
@end
// CHECK-OBJC_INTERFACE: ObjCCategoryDecl{{.*}} testCategory
// CHECK-OBJC_INTERFACE-NOT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_CATEGORY: ObjCCategoryDecl{{.*}} testCategory
// CHECK-OBJC_CATEGORY-NEXT: ObjCInterface
// CHECK-OBJC_CATEGORY-NEXT: AnnotateAttr{{.*}} "test"

// CHECK-OBJC_INTERFACE-LABEL: ObjCProtocolDecl
@protocol testProtocol
@end
// CHECK-OBJC_PROTOCOL: ObjCProtocolDecl{{.*}} testProtocol
// CHECK-OBJC_PROTOCOL-NEXT: AnnotateAttr{{.*}} "test"

@interface methodContainer
- (void) testInstanceMethod;
+ (void) testClassMethod;
@end
// CHECK-OBJC_METHOD: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-OBJC_METHOD-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-OBJC_METHOD-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD_IS_INSTANCE: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-OBJC_METHOD_IS_INSTANCE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD_IS_INSTANCE-LABEL: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-OBJC_METHOD_IS_INSTANCE-NOT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE-LABEL: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NOT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE-LABEL: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NOT: AnnotateAttr{{.*}} "test"

@implementation methodContainer
- (void) testInstanceMethod { }
+ (void) testClassMethod { }
@end
// CHECK-OBJC_METHOD: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-OBJC_METHOD-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD-NEXT: CompoundStmt
// CHECK-OBJC_METHOD-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-OBJC_METHOD-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD-NEXT: CompoundStmt
// CHECK-OBJC_METHOD-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD_IS_INSTANCE-LABEL: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-OBJC_METHOD_IS_INSTANCE-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD_IS_INSTANCE-NEXT: ImplicitParamDecl
// CHECK-OBJC_METHOD_IS_INSTANCE-NEXT: CompoundStmt
// CHECK-OBJC_METHOD_IS_INSTANCE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-OBJC_METHOD_IS_INSTANCE: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-OBJC_METHOD_IS_INSTANCE-NOT: AnnotateAttr{{.*}} "test"

// CHECK-HAS_TYPE_FUNCTION_TYPE-LABEL: ObjCMethodDecl{{.*}} testInstanceMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NOT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE-LABEL: ObjCMethodDecl{{.*}} testClassMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NOT: AnnotateAttr{{.*}} "test"
@interface propertyContainer {
  int testIvar;
// CHECK-FIELD: ObjCIvarDecl{{.*}} testIvar
// CHECK-FIELD-NEXT: AnnotateAttr{{.*}} "test"

}
@property int testProperty;
// CHECK-OBJC_PROPERTY: ObjCPropertyDecl{{.*}} testProperty
// CHECK-OBJC_PROPERTY-NEXT: AnnotateAttr{{.*}} "test"

@end

void (^testBlockVar)(void);
// CHECK-BLOCK: VarDecl{{.*}} testBlockVar
// CHECK-BLOCK-NOT: AnnotateAttr{{.*}} "test"

void testBlock(void) {
  (void)(^ { });
}
// CHECK-BLOCK-LABEL: BlockDecl
// CHECK-BLOCK-NEXT: CompoundStmt
// CHECK-BLOCK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE-LABEL: FunctionDecl{{.*}} testBlock
// CHECK-HAS_TYPE_FUNCTION_TYPE: BlockDecl
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: CompoundStmt
// The attribute applies to function, but not to block:
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE-NOT: AnnotateAttr{{.*}} "test"


#pragma clang attribute pop
