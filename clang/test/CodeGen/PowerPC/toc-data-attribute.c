// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -S -mtocdata=f,g,h,i,j,k,l,m,n,o,p -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=CHECK32 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -S -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=CHECK32 --match-full-lines

// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -S -mtocdata=f,g,h,i,j,k,l,m,n,o,p -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=CHECK64 --match-full-lines
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -S -mtocdata -emit-llvm -o - 2>&1 | FileCheck %s -check-prefixes=CHECK64 --match-full-lines

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
// CHECK32: {{.*}}  = private unnamed_addr constant [2 x i8] c"h\00", align 1
// CHECK32: @h = global {{...*}} #0
// CHECK32: @j = global i32 0, align 128
// CHECK32: @k = global float 1.000000e+02, align 4 #0
// CHECK32: @l = global double 2.500000e+00, align 8
// CHECK32: @m = global i32 10, section "foo", align 4
// CHECK32: @f = external global i32, align 4 #0
// CHECK32: @o = external global %struct.SomeStruct, align 1
// CHECK32: @i = global ptr null, align 4 #0
// CHECK32: @n = thread_local global i32 0, align 4
// CHECK32: @p = external global [0 x i32], align 4
// CHECK32: attributes #0 = { "toc-data" }

// CHECK64: @g = global i64 5, align 8 #0
// CHECK64: {{.*}}  = private unnamed_addr constant [2 x i8] c"h\00", align 1
// CHECK64: @h = global {{...*}} #0
// CHECK64: @j = global i32 0, align 128
// CHECK64: @k = global float 1.000000e+02, align 4 #0
// CHECK64: @l = global double 2.500000e+00, align 8 #0
// CHECK64: @m = global i32 10, section "foo", align 4
// CHECK64: @f = external global i32, align 4 #0
// CHECK64: @o = external global %struct.SomeStruct, align 1
// CHECK64: @i = global ptr null, align 8 #0
// CHECK64: @n = thread_local global i32 0, align 4
// CHECK64: @p = external global [0 x i32], align 4
// CHECK64: attributes #0 = { "toc-data" }
