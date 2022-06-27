// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter SomeClass -x objective-c++ | FileCheck %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter "(anonymous)" -x objective-c++ | FileCheck -check-prefix=CHECK-ANONYMOUS-ENUM %s

#import <CXXInteropKit/CXXInteropKit.h>

// CHECK: Dumping NSSomeClass:
// CHECK-NEXT: ObjCInterfaceDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> NSSomeClass
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeClass"

// CHECK: Dumping NSSomeClass::didMoveToParentViewController::
// CHECK-NEXT: ObjCMethodDecl {{.+}} imported in CXXInteropKit - didMoveToParentViewController: 'void'
// CHECK-NEXT: ParmVarDecl
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "didMove(toParent:)"

// CHECK: Dumping SomeClassRed:
// CHECK-NEXT: EnumConstantDecl {{.+}} imported in CXXInteropKit SomeClassRed 'ColorEnum'
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "red"

// CHECK-ANONYMOUS-ENUM: Dumping (anonymous):
// CHECK-ANONYMOUS-ENUM-NEXT: EnumDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> 'NSSomeEnumOptions':'unsigned long'
// CHECK-ANONYMOUS-ENUM-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeEnum.Options"
