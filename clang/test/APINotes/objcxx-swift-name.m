// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter SomeClass -x objective-c++ | FileCheck %s

#import <CxxInteropKit/CxxInteropKit.h>

// CHECK: Dumping NSSomeClass:
// CHECK-NEXT: ObjCInterfaceDecl {{.+}} imported in CxxInteropKit <undeserialized declarations> NSSomeClass
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeClass"

// CHECK: Dumping SomeClassRed:
// CHECK-NEXT: EnumConstantDecl {{.+}} imported in CxxInteropKit SomeClassRed 'ColorEnum'
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "red"