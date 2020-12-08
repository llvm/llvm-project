// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

__attribute__((swift_attr("@actor")))
@interface View
@end

// CHECK: InterfaceDecl {{.*}} View
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@actor"
