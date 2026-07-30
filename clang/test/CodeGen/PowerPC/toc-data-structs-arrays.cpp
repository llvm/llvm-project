// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -mtocdata=a4,a5,a8,a9,b,c,d,e,v -emit-llvm -o - 2>&1 \
// RUN:          | FileCheck %s -check-prefixes=CHECK32 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -mtocdata -emit-llvm -o - 2>&1 \
// RUN:          | FileCheck %s -check-prefixes=CHECK32 --match-full-lines

// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -mtocdata=a4,a5,a8,a9,b,c,d,e,v -emit-llvm -o - 2>&1 \
// RUN:          | FileCheck %s -check-prefixes=CHECK64 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -mtocdata -emit-llvm -o - 2>&1 \
// RUN:          | FileCheck %s -check-prefixes=CHECK64 --match-full-lines

struct size4_struct {
  int x;
};

struct size5_struct {
  int x;
  char c;
};

struct size8_struct {
  int x;
  short y;
  short z;
};

struct size9_struct {
  int x;
  short y;
  short z;
  char c;
};

struct size4_struct a4;
struct size5_struct a5;
struct size8_struct a8;
struct size9_struct a9;

short b[2];
short c[3];
short d[4];
short e[5];

int func_a() {
  return a4.x+a5.x+a8.x+a9.x+b[0]+c[0]+d[0]+e[0];
}

// CHECK32: @a4 = global %struct.size4_struct zeroinitializer, align 4 #0
// CHECK32: @a5 = global %struct.size5_struct zeroinitializer, align 4
// CHECK32: @a8 = global %struct.size8_struct zeroinitializer, align 4
// CHECK32: @a9 = global %struct.size9_struct zeroinitializer, align 4
// CHECK32: @b = global [2 x i16] zeroinitializer, align 2 #0
// CHECK32: @c = global [3 x i16] zeroinitializer, align 2
// CHECK32: @d = global [4 x i16] zeroinitializer, align 2
// CHECK32: @e = global [5 x i16] zeroinitializer, align 2
// CHECK32: attributes #0 = { "toc-data" }

// CHECK64: @a4 = global %struct.size4_struct zeroinitializer, align 4 #0
// CHECK64: @a5 = global %struct.size5_struct zeroinitializer, align 4 #0
// CHECK64: @a8 = global %struct.size8_struct zeroinitializer, align 4 #0
// CHECK64: @a9 = global %struct.size9_struct zeroinitializer, align 4
// CHECK64: @b = global [2 x i16] zeroinitializer, align 2 #0
// CHECK64: @c = global [3 x i16] zeroinitializer, align 2 #0
// CHECK64: @d = global [4 x i16] zeroinitializer, align 2 #0
// CHECK64: @e = global [5 x i16] zeroinitializer, align 2
// CHECK64: attributes #0 = { "toc-data" }
