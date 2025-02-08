// RUN: %clang_cc1 -std=c++20 -ast-dump -ast-dump-filter=pr126341 %s | FileCheck %s

template<_Complex int x>
struct pr126341;
template<>
struct pr126341<{1, 2}>;

// CHECK: Dumping pr126341:
// CHECK-NEXT: ClassTemplateDecl
// CHECK: Dumping pr126341:
// CHECK-NEXT: ClassTemplateSpecializationDecl
// CHECK-NEXT: `-TemplateArgument structural value '1+2i'
