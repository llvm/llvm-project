// RUN: %clang_cc1 %s -O0 -gline-tables-only -S -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" doesn't emit debug info
// for variables and types.

// CHECK-NOT: DW_TAG_namespace
namespace NS {
// CHECK-NOT: DW_TAG_class_type
// CHECK-NOT: DW_TAG_friend
class C { friend class D; };
class D {};
// CHECK-NOT: DW_TAG_inheritance
class E : public C {
  // CHECK-NOT: DW_TAG_reference type
  void x(const D& d);
};
}

// CHECK-NOT: DW_TAG_variable
NS::C c;
NS::D d;
NS::E e;
