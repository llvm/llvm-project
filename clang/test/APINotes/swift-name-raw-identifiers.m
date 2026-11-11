// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/RawIdentifiers -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/RawIdentifiers -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter NSSomeClass -x objective-c | FileCheck %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/RawIdentifiers -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter NSSomeEnumWith -x objective-c | FileCheck -check-prefix=CHECK-ENUM-CASE %s

#import <RawIdentifiers/RawIdentifiers.h>

// CHECK: Dumping NSSomeClass:
// CHECK-NEXT: ObjCInterfaceDecl {{.+}} imported in RawIdentifiers <undeserialized declarations> NSSomeClass
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "`Some Class`"

// CHECK: Dumping NSSomeClass::methodWithRawName::
// CHECK-NEXT: ObjCMethodDecl {{.+}} imported in RawIdentifiers - methodWithRawName: 'void'
// CHECK-NEXT: ParmVarDecl
// CHECK-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "`raw method`(`raw param`:)"

// CHECK-ENUM-CASE: Dumping NSSomeEnumWithRed:
// CHECK-ENUM-CASE-NEXT: EnumConstantDecl {{.+}} imported in RawIdentifiers NSSomeEnumWithRed 'int'
// CHECK-ENUM-CASE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "red"

// CHECK-ENUM-CASE: Dumping NSSomeEnumWithRawName:
// CHECK-ENUM-CASE-NEXT: EnumConstantDecl {{.+}} imported in RawIdentifiers NSSomeEnumWithRawName 'int'
// CHECK-ENUM-CASE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "`raw constant`"
