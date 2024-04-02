// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -S -mtocdata=f,g,h,i,j,k,l,m,n,o,p -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,CHECK32 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -S -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,CHECK32 --match-full-lines

// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -S -mtocdata=f,g,h,i,j,k,l,m,n,o,p -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,CHECK64 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -S -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=COMMON,CHECK64 --match-full-lines

extern int f;
long long g = 5;
const char *h = "h";
int *i;
int __attribute__((aligned(128))) j = 0;
float k = 100.00;
double l = 2.5;
int m __attribute__((section("foo"))) = 10;
__thread int n;

extern int p[];

struct SomeStruct;
extern struct SomeStruct o;

static int func_a() {
  return g+(int)h[0]+*i+j+k+l+m+n+p[0];
}

int func_b() {
  f = 1;
  return func_a();
}

struct SomeStruct* getAddress(void) {
  return &o;
}

// CHECK32: @g = global i64 5, align 8
// CHECK64: @g = global i64 5, align 8 #0
// COMMON: {{.*}}  = private unnamed_addr constant [2 x i8] c"h\00", align 1
// COMMON: @h = global {{...*}} #0
// COMMON: @j = global i32 0, align 128
// COMMON: @k = global float 1.000000e+02, align 4 #0
// CHECK32: @l = global double 2.500000e+00, align 8
// CHECK64: @l = global double 2.500000e+00, align 8 #0
// COMMON: @m = global i32 10, section "foo", align 4
// COMMON: @f = external global i32, align 4 #0
// COMMON: @o = external global %struct.SomeStruct, align 1
// CHECK32: @i = global ptr null, align 4 #0
// CHECK64: @i = global ptr null, align 8 #0
// COMMON: @n = thread_local global i32 0, align 4
// COMMON: @p = external global [0 x i32], align 4
// COMMON: attributes #0 = { "toc-data" }
