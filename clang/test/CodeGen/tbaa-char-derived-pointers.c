// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OLD
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -disable-llvm-passes -emit-llvm -new-struct-path-tbaa -o - %s | FileCheck %s --check-prefixes=CHECK,NEW
//
// char-derived pointers must use the generic "any pN pointer" TBAA node, not a
// distinct "pN omnipotent char" sibling of "pN int" (which would imply NoAlias).

void char_int_ptr_cast(int ***p) {
  char ***c = (char ***)p;
  *c = 0;
  (void)*p;
}

// CHECK-LABEL: define dso_local void @char_int_ptr_cast(
// CHECK: store ptr null, ptr {{.*}}, !tbaa !{{[0-9]+}}

// OLD-DAG: !{{[0-9]+}} = !{!"p2 int", !{{[0-9]+}}, i64 0}
// OLD-DAG: !{{[0-9]+}} = !{!"any p2 pointer", !{{[0-9]+}}, i64 0}

// NEW-DAG: !{{[0-9]+}} = !{!{{[0-9]+}}, i64 8, !"p2 int"}
// NEW-DAG: !{{[0-9]+}} = !{!{{[0-9]+}}, i64 8, !"any p2 pointer"}

// CHECK-NOT: !{!"p2 omnipotent char"
// CHECK-NOT: !{!"p3 omnipotent char"
