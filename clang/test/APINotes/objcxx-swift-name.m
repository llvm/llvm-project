// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter SomeClass -x objective-c++ | FileCheck %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter "unnamed enum" -x objective-c++ | FileCheck -check-prefix=CHECK-ANONYMOUS-ENUM %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::varInNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-GLOBAL-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::funcInNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-FUNC-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::varInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-GLOBAL-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested2::varInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::funcInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-FUNC-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::Namespace1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE %s

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

// CHECK-ANONYMOUS-ENUM: Dumping (unnamed enum at {{.+}}/Frameworks/CXXInteropKit.framework/Headers/CXXInteropKit.h:19:9):
// CHECK-ANONYMOUS-ENUM-NEXT: EnumDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> 'NSSomeEnumOptions':'unsigned long'
// CHECK-ANONYMOUS-ENUM-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SomeEnum.Options"

// CHECK-GLOBAL-IN-NAMESPACE: Dumping Namespace1::varInNamespace:
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: VarDecl {{.+}} imported in CXXInteropKit varInNamespace 'int' static cinit
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 1
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftVarInNamespace"

// CHECK-FUNC-IN-NAMESPACE: Dumping Namespace1::funcInNamespace:
// CHECK-FUNC-IN-NAMESPACE-NEXT: FunctionDecl {{.+}} imported in CXXInteropKit funcInNamespace 'void ()'
// CHECK-FUNC-IN-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftFuncInNamespace()"

// CHECK-STRUCT-IN-NAMESPACE: Dumping Namespace1::char_box:
// CHECK-STRUCT-IN-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "CharBox"

// CHECK-GLOBAL-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::varInNestedNamespace:
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: VarDecl {{.+}} imported in CXXInteropKit varInNestedNamespace 'int' static cinit
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 1
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftVarInNestedNamespace"

// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested2::varInNestedNamespace:
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: VarDecl {{.+}} imported in CXXInteropKit varInNestedNamespace 'int' static cinit
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 2
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftAnotherVarInNestedNamespace"

// CHECK-FUNC-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::funcInNestedNamespace:
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: FunctionDecl {{.+}} imported in CXXInteropKit funcInNestedNamespace 'void (int)'
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: ParmVarDecl {{.+}} i 'int'
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftFuncInNestedNamespace(_:)"

// CHECK-STRUCT-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::char_box:
// CHECK-STRUCT-IN-NESTED-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-NESTED-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "NestedCharBox"

// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE: Dumping Namespace1::Nested1::Namespace1::char_box:
// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in CXXInteropKit <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "DeepNestedCharBox"
