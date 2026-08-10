// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -Wno-objc-root-class -ast-dump -ast-dump-filter test %s | FileCheck %s

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = any(objc_interface, objc_protocol, objc_property, field, objc_method, variable))
#pragma clang attribute push (__attribute__((objc_subclassing_restricted)), apply_to = objc_interface)

@interface testInterface1
// CHECK-LABEL: ObjCInterfaceDecl{{.*}}testInterface1
// CHECK-NEXT: ObjCImplementation
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: ObjCSubclassingRestrictedAttr{{.*}}

// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

{
  int testIvar1;
  // CHECK-LABEL: ObjCIvarDecl{{.*}} testIvar1
  // CHECK-NEXT: AnnotateAttr{{.*}} "test"
  // CHECK-NOT: ObjCSubclassingRestrictedAttr
}

@property int testProp1;
// CHECK-LABEL: ObjCPropertyDecl{{.*}} testProp1
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr

- (void)testIm:(int) x;
// CHECK-LABEL: ObjCMethodDecl{{.*}}testIm
// CHECK-NEXT: ParmVarDecl{{.*}} x
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr

+ (void)testCm;
// CHECK-LABEL: ObjCMethodDecl{{.*}}testCm
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr

// Implicit getters/setters shouldn't receive the attributes.
// CHECK-LABEL: ObjCMethodDecl{{.*}}testProp1
// CHECK-NOT: AnnotateAttr
// CHECK-LABEL: ObjCMethodDecl{{.*}}setTestProp1
// CHECK-NOT: AnnotateAttr

@end

// @implementation can't receive explicit attributes, so don't add the pragma
// attributes to them.
@implementation testInterface1
// CHECK-LABEL: ObjCImplementationDecl{{.*}}testInterface1
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

{
  int testIvar2;
  // CHECK-LABEL: ObjCIvarDecl{{.*}} testIvar2
  // CHECK-NEXT: AnnotateAttr{{.*}} "test"
  // CHECK-NOT: ObjCSubclassingRestrictedAttr
}

// Don't add attributes to implicit parameters!
- (void)testIm:(int) x {
// CHECK-LABEL: ObjCMethodDecl{{.*}}testIm
// CHECK-NEXT: ImplicitParamDecl
// CHECK-NEXT: ImplicitParamDecl
// CHECK-NEXT: ParmVarDecl{{.*}} x
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr
}

+ (void)testCm {
// CHECK-LABEL: ObjCMethodDecl{{.*}}testCm
// CHECK: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr
// CHECK-NOT: AnnotateAttr
  _Pragma("clang attribute push (__attribute__((annotate(\"applied at container start\"))), apply_to=objc_interface)");
}

// Implicit ivars shouldn't receive the attributes.
// CHECK-LABEL: ObjCIvarDecl{{.*}}_testProp1
// CHECK-NOT: AnnotateAttr

@end

@implementation testImplWithoutInterface // expected-warning {{cannot find interface declaration for 'testImplWithoutInterface'}}
// CHECK-LABEL: ObjCInterfaceDecl{{.*}}testImplWithoutInterface
// CHECK-NEXT: ObjCImplementation
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: ObjCSubclassingRestrictedAttr
// CHECK-NEXT: AnnotateAttr{{.*}} "applied at container start"

// CHECK-LABEL: ObjCImplementationDecl{{.*}}testImplWithoutInterface
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

@end

#pragma clang attribute pop

@protocol testProtocol
// CHECK-LABEL: ObjCProtocolDecl{{.*}}testProtocol
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr
// CHECK-NOT: AnnotateAttr

- (void)testProtIm;
// CHECK-LABEL: ObjCMethodDecl{{.*}}testProtIm
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr

@end

@protocol testForwardProtocol;
// CHECK-LABEL: ObjCProtocolDecl{{.*}}testForwardProtocol
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr


// Categories can't receive explicit attributes, so don't add pragma attributes
// to them.
@interface testInterface1(testCat)
// CHECK-LABEL: ObjCCategoryDecl{{.*}}testCat
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

@end

@implementation testInterface1(testCat)
// CHECK-LABEL: ObjCCategoryImplDecl{{.*}}testCat
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

@end

// @class/@compatibility_alias declarations can't receive explicit attributes,
// so don't add pragma attributes to them.
@class testClass;
// CHECK-LABEL: ObjCInterfaceDecl{{.*}}testClass
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

@compatibility_alias testCompat testInterface1;
// CHECK-LABEL: ObjCCompatibleAliasDecl{{.*}}testCompat
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr

#pragma clang attribute pop // objc_subclassing_restricted

@interface testInterface3
// CHECK-LABEL: ObjCInterfaceDecl{{.*}}testInterface3
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NOT: ObjCSubclassingRestrictedAttr
@end

#pragma clang attribute pop // annotate("test")

@interface testInterface4
// CHECK-LABEL: ObjCInterfaceDecl{{.*}}testInterface4
// CHECK-NOT: AnnotateAttr
// CHECK-NOT: ObjCSubclassingRestrictedAttr
@end
