// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

__attribute__((swift_attr("@actor")))
@interface View
@end

// CHECK-LABEL: InterfaceDecl {{.*}} View
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@actor"

#pragma clang attribute push(__attribute__((swift_attr("@sendable"))), apply_to=objc_interface)
@interface Contact
@end
#pragma clang attribute pop

// CHECK-LABEL: InterfaceDecl {{.*}} Contact
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@sendable"
