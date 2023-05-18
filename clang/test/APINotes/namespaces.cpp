// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x objective-c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::my_typedef -x objective-c++ | FileCheck -check-prefix=CHECK-TYPEDEF-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::my_using_decl -x objective-c++ | FileCheck -check-prefix=CHECK-USING-DECL-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::varInNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-GLOBAL-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::funcInNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-FUNC-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::varInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-GLOBAL-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested2::varInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::funcInNestedNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-FUNC-IN-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Namespace1::Nested1::Namespace1::char_box -x objective-c++ | FileCheck -check-prefix=CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter varInInlineNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-GLOBAL-IN-INLINE-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/CxxInterop -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter funcInInlineNamespace -x objective-c++ | FileCheck -check-prefix=CHECK-FUNC-IN-INLINE-NAMESPACE %s

#import <Namespaces.h>

// CHECK-TYPEDEF-IN-NAMESPACE: Dumping Namespace1::my_typedef:
// CHECK-TYPEDEF-IN-NAMESPACE-NEXT: TypedefDecl {{.+}} imported in Namespaces my_typedef 'int'
// CHECK-TYPEDEF-IN-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "SwiftTypedef"

// CHECK-USING-DECL-IN-NAMESPACE: Dumping Namespace1::my_using_decl:
// CHECK-USING-DECL-IN-NAMESPACE-NEXT: TypeAliasDecl {{.+}} imported in Namespaces my_using_decl 'int'
// CHECK-USING-DECL-IN-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "SwiftUsingDecl"

// CHECK-GLOBAL-IN-NAMESPACE: Dumping Namespace1::varInNamespace:
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: VarDecl {{.+}} imported in Namespaces varInNamespace 'int' static cinit
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 1
// CHECK-GLOBAL-IN-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftVarInNamespace"

// CHECK-FUNC-IN-NAMESPACE: Dumping Namespace1::funcInNamespace:
// CHECK-FUNC-IN-NAMESPACE-NEXT: FunctionDecl {{.+}} imported in Namespaces funcInNamespace 'void ()'
// CHECK-FUNC-IN-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftFuncInNamespace()"

// CHECK-STRUCT-IN-NAMESPACE: Dumping Namespace1::char_box:
// CHECK-STRUCT-IN-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in Namespaces <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "CharBox"

// CHECK-GLOBAL-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::varInNestedNamespace:
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: VarDecl {{.+}} imported in Namespaces varInNestedNamespace 'int' static cinit
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 1
// CHECK-GLOBAL-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftVarInNestedNamespace"

// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested2::varInNestedNamespace:
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: VarDecl {{.+}} imported in Namespaces varInNestedNamespace 'int' static cinit
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 2
// CHECK-ANOTHER-GLOBAL-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftAnotherVarInNestedNamespace"

// CHECK-FUNC-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::funcInNestedNamespace:
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: FunctionDecl {{.+}} imported in Namespaces funcInNestedNamespace 'void (int)'
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: ParmVarDecl {{.+}} i 'int'
// CHECK-FUNC-IN-NESTED-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftFuncInNestedNamespace(_:)"

// CHECK-STRUCT-IN-NESTED-NAMESPACE: Dumping Namespace1::Nested1::char_box:
// CHECK-STRUCT-IN-NESTED-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in Namespaces <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-NESTED-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "NestedCharBox"

// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE: Dumping Namespace1::Nested1::Namespace1::char_box:
// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE-NEXT: CXXRecordDecl {{.+}} imported in Namespaces <undeserialized declarations> struct char_box
// CHECK-STRUCT-IN-DEEP-NESTED-NAMESPACE: SwiftNameAttr {{.+}} <<invalid sloc>> "DeepNestedCharBox"

// CHECK-GLOBAL-IN-INLINE-NAMESPACE: Dumping varInInlineNamespace:
// CHECK-GLOBAL-IN-INLINE-NAMESPACE-NEXT: VarDecl {{.+}} imported in Namespaces varInInlineNamespace 'int' static cinit
// CHECK-GLOBAL-IN-INLINE-NAMESPACE-NEXT: IntegerLiteral {{.+}} 'int' 3
// CHECK-GLOBAL-IN-INLINE-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftVarInInlineNamespace"

// CHECK-FUNC-IN-INLINE-NAMESPACE: Dumping funcInInlineNamespace:
// CHECK-FUNC-IN-INLINE-NAMESPACE-NEXT: FunctionDecl {{.+}} imported in Namespaces funcInInlineNamespace 'void ()'
// CHECK-FUNC-IN-INLINE-NAMESPACE-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "swiftFuncInInlineNamespace()"
