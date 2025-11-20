// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// Capture the anonymous struct types for check lines below.
// CHECK: [[ANON_1:%.*]] = type <{ float, target("dx.Padding", 12), <4 x i32> }>
// CHECK: [[ANON_2:%.*]] = type <{ <2 x i32>, target("dx.Padding", 8), <{ [3 x <{ %ArrayAndScalar, target("dx.Padding", 12) }>], %ArrayAndScalar }>

template <typename T> void use(T);

cbuffer CBArrays : register(b2) {
  float c1[30];
  double3 c2[20];
  float16_t c3[10][20];
  uint64_t c4[30];
  int4 c5[20][30][40];
  uint16_t c6[10];
  int64_t c7[20];
  bool c8[40];
}

// CHECK-LABEL: define hidden void @_Z8cbarraysv()
void cbarrays() {
  // CHECK: load float, ptr addrspace(2) @c1, align 16
  use(c1[0]);
  // CHECK: load float, ptr addrspace(2) getelementptr (<{ float, target("dx.Padding", 12) }>, ptr addrspace(2) @c1, i32 7, i32 0), align 16
  use(c1[7]);
  // CHECK: load float, ptr addrspace(2) getelementptr (<{ float, target("dx.Padding", 12) }>, ptr addrspace(2) @c1, i32 29, i32 0), align 16
  use(c1[29]);

  // CHECK: load <3 x double>, ptr addrspace(2) getelementptr (<{ <3 x double>, target("dx.Padding", 8) }>, ptr addrspace(2) @c2, i32 8, i32 0), align 32
  use(c2[8]);
  // CHECK: load half, ptr addrspace(2) getelementptr (<{ half, target("dx.Padding", 14) }>, ptr addrspace(2) getelementptr (<{ <{ [19 x <{ half, target("dx.Padding", 14) }>], half }>, target("dx.Padding", 14) }>, ptr addrspace(2) @c3, i32 9, i32 0), i32 5, i32 0), align 16
  use(c3[9][5]);
  // CHECK: load i64, ptr addrspace(2) getelementptr (<{ i64, target("dx.Padding", 8) }>, ptr addrspace(2) @c4, i32 6, i32 0), align 16
  use(c4[6]);
  // CHECK:  load <4 x i32>, ptr addrspace(2) getelementptr inbounds ([40 x <4 x i32>], ptr addrspace(2) getelementptr inbounds ([30 x [40 x <4 x i32>]], ptr addrspace(2) getelementptr inbounds ([20 x [30 x [40 x <4 x i32>]]], ptr addrspace(2) @c5, i32 0, i32 1), i32 0, i32 12), i32 0, i32 15), align 16
  use(c5[1][12][15]);
  // CHECK: load i16, ptr addrspace(2) getelementptr (<{ i16, target("dx.Padding", 14) }>, ptr addrspace(2) @c6, i32 4, i32 0), align 16
  use(c6[4]);
  // CHECK: load i64, ptr addrspace(2) getelementptr (<{ i64, target("dx.Padding", 8) }>, ptr addrspace(2) @c7, i32 17, i32 0), align 16
  use(c7[17]);
  // CHECK: load i32, ptr addrspace(2) getelementptr (<{ i32, target("dx.Padding", 12) }>, ptr addrspace(2) @c8, i32 30, i32 0), align 16
  use(c8[30]);
}

struct A {
  float2 a1;
};

struct B : A {
  uint16_t3 b1;
};

struct C {
  int c1;
  A c2;
};

struct D {
  B d1[4][6];
};

cbuffer CBStructs {
  A s1;
  B s2;
  C s3;
  A s4[5];
  D s5;
};

// CHECK-LABEL: define hidden void @_Z9cbstructsv()
void cbstructs() {
  // CHECK: load <2 x float>, ptr addrspace(2) @s1, align 8
  use(s1.a1);
  // CHECK: load <3 x i16>, ptr addrspace(2) getelementptr inbounds nuw (%B, ptr addrspace(2) @s2, i32 0, i32 1), align 2
  use(s2.b1);
  // CHECK: load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (%C, ptr addrspace(2) @s3, i32 0, i32 1), align 8
  use(s3.c2.a1);
  // CHECK: load <2 x float>, ptr addrspace(2) getelementptr (<{ %A, target("dx.Padding", 8) }>, ptr addrspace(2) @s4, i32 2, i32 0), align 8
  use(s4[2].a1);
  // CHECK: load <3 x i16>, ptr addrspace(2) getelementptr inbounds nuw (%B, ptr addrspace(2) getelementptr (<{ %B, target("dx.Padding", 2) }>, ptr addrspace(2) getelementptr (<{ <{ [5 x <{ %B, target("dx.Padding", 2) }>], %B }>, target("dx.Padding", 2) }>, ptr addrspace(2) @s5, i32 3, i32 0), i32 5, i32 0), i32 0, i32 1), align 2
  use(s5.d1[3][5].b1);
}

struct Scalars {
  float a, b;
};

struct ArrayAndScalar {
  uint4 x[5];
  float y;
};

cbuffer CBMix {
  Scalars m1[3];
  float m2;
  ArrayAndScalar m3;
  float2 m4[5][4];
  struct { float c; uint4 d; } m5;
  struct { int2 i; ArrayAndScalar j[4]; } m6;
  vector<double, 1> m7;
};

// CHECK-LABEL: define hidden void @_Z5cbmixv()
void cbmix() {
  // CHECK: load float, ptr addrspace(2) getelementptr inbounds nuw (%Scalars, ptr addrspace(2) getelementptr (<{ %Scalars, target("dx.Padding", 8) }>, ptr addrspace(2) @m1, i32 2, i32 0), i32 0, i32 1), align 4
  use(m1[2].b);
  // CHECK: load float, ptr addrspace(2) getelementptr inbounds nuw (%ArrayAndScalar, ptr addrspace(2) @m3, i32 0, i32 1), align 4
  use(m3.y);
  // CHECK: load <2 x float>, ptr addrspace(2) getelementptr (<{ <2 x float>, target("dx.Padding", 8) }>, ptr addrspace(2) getelementptr (<{ <{ [3 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, target("dx.Padding", 8) }>, ptr addrspace(2) @m4, i32 2, i32 0), i32 3, i32 0), align 16
  use(m4[2][3]);
  // CHECK: load <4 x i32>, ptr addrspace(2) getelementptr inbounds nuw ([[ANON_1]], ptr addrspace(2) @m5, i32 0, i32 1), align 16
  use(m5.d);
  // CHECK: load <4 x i32>, ptr addrspace(2) getelementptr inbounds ([5 x <4 x i32>], ptr addrspace(2) getelementptr (<{ %ArrayAndScalar, target("dx.Padding", 12) }>, ptr addrspace(2) getelementptr inbounds nuw ([[ANON_2]], ptr addrspace(2) @m6, i32 0, i32 1), i32 2, i32 0), i32 0, i32 2), align 16
  use(m6.j[2].x[2]);
  // CHECK: load <1 x double>, ptr addrspace(2) @m7, align 8
  use(m7);
}
