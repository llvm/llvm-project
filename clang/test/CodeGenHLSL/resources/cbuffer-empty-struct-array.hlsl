// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes \
// RUN:     -finclude-default-header -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes \
// RUN:     -finclude-default-header -o - -DCHECK2 %s | FileCheck %s -check-prefix=CHECK2

// Regression test for issue llvm/llvm-project#183788

// empty struct
struct A {
};

// struct with a resource which does not contribute to cbuffer layout
struct B {
    RWBuffer<float> Buf;
};

struct C : A {
  B b;
  RWBuffer<float> Bufs[10];
  A array[10][2];
};

cbuffer CB {
  A a[2];
  C c;
#ifdef CHECK2
  int i;
  float2 v;
  int4x4 m;
#endif
}

B b[2][2];

#ifdef CHECK2
int j;
float2 w; 
int4x4 n;
#endif

[numthreads(4,4,4)]
void main() {
}

// CHECK-NOT: @CB.cb = global target("dx.CBuffer", %__cblayout_CB)
// CHECK-NOT: @a = external hidden addrspace(2) global
// CHECK-NOT: @b = external hidden addrspace(2) global
// CHECK-NOT: @c = external hidden addrspace(2) global
// CHECK-NOT: @"$Globals.cb" = global target("dx.CBuffer",

// CHECK2: @CB.cb = global target("dx.CBuffer", %__cblayout_CB)
// CHECK-NOT: @a = external hidden addrspace(2) global
// CHECK-NOT: @c = external hidden addrspace(2) global
// CHECK2: @i = external hidden addrspace(2) global i32
// CHECK2: @v = external hidden addrspace(2) global <2 x float>, align 8
// CHECK2: @m = external hidden addrspace(2) global [4 x <4 x i32>], align 4
// CHECK2: @"$Globals.cb" = global target("dx.CBuffer",
// CHECK-NOT: @b = external hidden addrspace(2) global
// CHECK2: @j = external hidden addrspace(2) global i32
// CHECK2: @w = external hidden addrspace(2) global <2 x float>, align 8
// CHECK2: @n = external hidden addrspace(2) global [4 x <4 x i32>], align 4
