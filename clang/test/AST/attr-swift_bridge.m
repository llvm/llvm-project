// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

struct __attribute__((__swift_bridge__("BridgedS"))) S;
// CHECK: RecordDecl {{.*}} struct S
// CHECK: SwiftBridgeAttr {{.*}} "BridgedS"

struct S {
};

// CHECK: RecordDecl {{.*}} struct S definition
// CHECK: SwiftBridgeAttr {{.*}} Inherited "BridgedS"
