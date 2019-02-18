// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s
// expected-no-diagnostics

__attribute__ ((external_source_symbol(language= "Swift", defined_in="A")))
@interface TestInterface
@end
// CHECK: ObjCInterfaceDecl {{.*}} TestInterface
// CHECK-NEXT: ExternalSourceSymbolAttr

__attribute__ ((external_source_symbol(language= "Swift", defined_in="B")))
@interface TestInterface ()
@end
// CHECK: ObjCCategoryDecl
// CHECK-NEXT: ObjCInterface
// CHECK-NEXT: ExternalSourceSymbolAttr {{.*}} "Swift" "B"

__attribute__ ((external_source_symbol(language= "Swift", defined_in="C")))
@interface TestInterface (Category)
@end
// CHECK: ObjCCategoryDecl
// CHECK-NEXT: ObjCInterface
// CHECK-NEXT: ExternalSourceSymbolAttr {{.*}} "Swift" "C"
