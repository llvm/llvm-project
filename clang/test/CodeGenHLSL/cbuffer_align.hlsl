// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-HALF

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-FLOAT


struct S1 {
  half a;   // 2 bytes + 2 bytes pad or 4 bytes
  float b;  // 4 bytes
  half c;   // 2 bytes + 2 bytes pad or 4 bytes
  float d;  // 4 bytes
  double e; // 8 bytes
};

struct S2 {
  half a;   // 2 bytes or 4 bytes
  half b;   // 2 bytes or 4 bytes
  float e;  // 4 bytes or 4 bytes + 4 padding
  double f; // 8 bytes
};

struct S3 {
  half a;     // 2 bytes + 6 bytes pad or 4 bytes + 4 bytes pad
  uint64_t b; // 8 bytes
};

struct S4 {
  float a;  // 4 bytes
  half b;   // 2 bytes or 4 bytes
  half c;   // 2 bytes or 4 bytes + 4 bytes pad
  double d; // 8 bytes
};


cbuffer CB0 {
  // CHECK-HALF: @CB0.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB0, 24, 0))
  // CHECK-FLOAT: @CB0.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB0, 24, 0))
  S1 s1;
}

cbuffer CB1 {
  // CHECK-HALF: @CB1.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB1, 16, 0))
  // CHECK-FLOAT: @CB1.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB1, 24, 0))
  S2 s2;
}

cbuffer CB2 {
  // CHECK-HALF: @CB2.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 16, 0))
  // CHECK-FLOAT: @CB2.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 16, 0))
  S3 s3;
}

cbuffer CB3 {
  // CHECK-HALF: @CB3.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB3, 16, 0))
  // CHECK-FLOAT: @CB3.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB3, 24, 0))
  S4 s4;
}
