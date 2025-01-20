// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -o - -O0 | FileCheck %s

typedef unsigned char uchar4 __attribute((ext_vector_type(4)));
typedef unsigned int int4 __attribute((ext_vector_type(4)));
typedef float float4 __attribute((ext_vector_type(4)));

// CHECK-LABEL: define{{.*}} spir_kernel void @ker()
void kernel ker() {
  bool t = true;
  int4 vec4 = (int4)t;
// CHECK: {{%.*}} = load i8, ptr %t, align 1
// CHECK: {{%.*}} = trunc i8 {{%.*}} to i1
// CHECK: {{%.*}} = sext i1 {{%.*}} to i32
// CHECK: {{%.*}} = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: {{%.*}} = shufflevector <4 x i32> {{%.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: store <4 x i32> {{%.*}}, ptr %vec4, align 16
  int i = (int)t;
// CHECK: {{%.*}} = load i8, ptr %t, align 1
// CHECK: {{%.*}} = trunc i8 {{%.*}} to i1
// CHECK: {{%.*}} = zext i1 {{%.*}} to i32
// CHECK: store i32 {{%.*}}, ptr %i, align 4

  uchar4 vc;
  vc = (uchar4)true;
// CHECK: store <4 x i8> splat (i8 -1), ptr %vc, align 4
  unsigned char c;
  c = (unsigned char)true;
// CHECK: store i8 1, ptr %c, align 1

  float4 vf;
  vf = (float4)true;
// CHECK: store <4 x float> splat (float -1.000000e+00)
}
