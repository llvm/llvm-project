// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

typedef int T1 __attribute__((__swift_newtype__(struct)));
typedef int T2 __attribute__((__swift_newtype__(enum)));

typedef int T3 __attribute__((__swift_wrapper__(struct)));
typedef int T4 __attribute__((__swift_wrapper__(enum)));

typedef int T5;
typedef int T5 __attribute__((__swift_wrapper__(struct)));
typedef int T5;
// CHECK-LABEL: TypedefDecl {{.+}} T5 'int'
// CHECK-NEXT: BuiltinType {{.+}} 'int'
// CHECK-NEXT: TypedefDecl {{.+}} T5 'int'
// CHECK-NEXT: BuiltinType {{.+}} 'int'
// CHECK-NEXT: SwiftNewTypeAttr {{.+}} NK_Struct
// CHECK-NEXT: TypedefDecl {{.+}} T5 'int'
// CHECK-NEXT: BuiltinType {{.+}} 'int'
// CHECK-NEXT: SwiftNewTypeAttr {{.+}} NK_Struct
