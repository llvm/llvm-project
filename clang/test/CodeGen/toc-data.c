// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -S -mtocdata=g1,g2,g3 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-MIX --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc64-unkown-aix -S -mtocdata -mno-tocdata=g4,g5 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-MIX --match-full-lines

// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -S -mno-tocdata -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-NOTOCDATA
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -S -mno-tocdata -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-NOTOCDATA

// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -S -mtocdata -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-TOCDATA --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -S -mtocdata -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-TOCDATA --match-full-lines

int g1, g4;
extern int g2;
int g3 = 0, g5 = 123;
void func() {
  g2 = 0;
}

// CHECK-MIX-DAG: @g3 = global i32 0, align 4 #0
// CHECK-MIX-DAG: @g2 = external global i32, align 4 #0
// CHECK-MIX-DAG: @g1 = global i32 0, align 4 #0
// CHECK-MIX-DAG: @g4 = global i32 0, align 4
// CHECK-MIX-DAG: @g5 = global i32 123, align 4
// CHECK-MIX: attributes #0 = { "toc-data" }

// CHECK-NOTOCDATA-NOT: "toc-data"

// CHECK-TOCDATA-DAG: @g3 = global i32 0, align 4 #0
// CHECK-TOCDATA-DAG: @g2 = external global i32, align 4 #0
// CHECK-TOCDATA-DAG: @g1 = global i32 0, align 4 #0
// CHECK-TOCDATA-DAG: @g4 = global i32 0, align 4 #0
// CHECK-TOCDATA-DAG: @g5 = global i32 123, align 4 #0
// CHECK-TOCDATA: attributes #0 = { "toc-data" }
