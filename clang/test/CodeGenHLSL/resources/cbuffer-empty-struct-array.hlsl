// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - -DCHECK2 %s | FileCheck %s -check-prefix=CHECK2

// Regression test for issue llvm/llvm-project#183788

// empty struct
struct A {
};

// struct with a resource which does not contribute to cbuffer layout
struct B {
    RWBuffer<float> Buf;
};

cbuffer CB {
  A a[2];
#ifdef CHECK2
  int i;
#endif
}

B b[2][2];
#ifdef CHECK2
int j;
#endif

[numthreads(4,4,4)]
void main() {
}

// CHECK-NOT: @CB.cb = global target("dx.CBuffer", %__cblayout_CB)
// CHECK-NOT: @A = external hidden addrspace(2) global
// CHECK-NOT: @B = external hidden addrspace(2) global
// CHECK-NOT: @"$Globals.cb" = global target("dx.CBuffer",

// CHECK2: @CB.cb = global target("dx.CBuffer", %__cblayout_CB)
// CHECK2-NOT: @A = external hidden addrspace(2) global
// CHECK2: @i = external hidden addrspace(2) global i32
// CHECK2: @"$Globals.cb" = global target("dx.CBuffer",
// CHECK2-NOT: @B = external hidden addrspace(2) global
// CHECK2: @j = external hidden addrspace(2) global i32
