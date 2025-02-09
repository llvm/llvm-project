// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

typedef struct T TBridged __attribute((__swift_bridged_typedef__));
// CHECK: TypedefDecl {{.*}} TBridged 'struct T'
// CHECK: SwiftBridgedTypedefAttr

typedef struct T TBridged;
// CHECK: TypedefDecl {{.*}} TBridged 'struct T'
// CHECK: SwiftBridgedTypedefAttr
