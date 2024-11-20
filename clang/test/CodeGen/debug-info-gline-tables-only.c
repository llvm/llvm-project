// RUN: %clang_cc1 %s -debug-info-kind=line-tables-only -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -debug-info-kind=line-directives-only -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" or "-gline-directives-only" doesn't emit debug info
// for variables and types.

// CHECK-NOT: DW_TAG_variable
int global = 42;

// CHECK-NOT: DW_TAG_typedef
// CHECK-NOT: DW_TAG_const_type
// CHECK-NOT: DW_TAG_pointer_type
// CHECK-NOT: DW_TAG_array_type
typedef const char* constCharPtrArray[10];

// CHECK-NOT: DW_TAG_structure_type
struct S {
  // CHECK-NOT: DW_TAG_member
  char a;
  double b;
  constCharPtrArray c;
};

// CHECK-NOT: DW_TAG_enumerator
// CHECK-NOT: DW_TAG_enumeration_type
enum E { ZERO = 0, ONE = 1 };

// CHECK-NOT: DILocalVariable
int sum(int p, int q) {
  int r = p + q;
  struct S s;
  enum E e;
  return r;
}
