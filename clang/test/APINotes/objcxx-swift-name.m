// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter SomeClass -x objective-c++ | FileCheck %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter method -x objective-c++ | FileCheck -check-prefix=CHECK-METHOD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter "(anonymous)" -x objective-c++ | FileCheck -check-prefix=CHECK-ANONYMOUS-ENUM %s

#import <CXXInteropKit/CXXInteropKit.h>

// CHECK: Dumping NSSomeClass:
// CHECK: ObjCInterfaceDecl {{.+}} imported in CxxInteropKit <undeserialized declarations> NSSomeClass
// CHECK: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeClass"

// CHECK: Dumping NSSomeClass::didMoveToParentViewController::
// CHECK: ObjCMethodDecl {{.+}} imported in CxxInteropKit - didMoveToParentViewController: 'void'
// CHECK: SwiftNameAttr {{.+}} <<invalid sloc>> "didMove(toParent:)"

// CHECK: Dumping SomeClassRed:
// CHECK: EnumConstantDecl {{.+}} imported in CxxInteropKit SomeClassRed 'ColorEnum'
// CHECK: SwiftNameAttr {{.+}} <<invalid sloc>> "red"

// CHECK-METHOD: Dumping GlobalStruct::method:
// CHECK-METHOD: SwiftNameAttr {{.+}} <<invalid sloc>> "globalMethod()"

// CHECK-METHOD: Dumping ParentNS::ChildNS::ParentStruct::ChildStruct::method:
// CHECK-METHOD: SwiftNameAttr {{.+}} <<invalid sloc>> "nestedMethod()"

// CHECK-ANONYMOUS-ENUM: Dumping (anonymous):
// CHECK-ANONYMOUS-ENUM-NEXT: EnumDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> 'NSSomeEnumOptions':'unsigned long'
// CHECK-ANONYMOUS-ENUM-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeEnum.Options"
