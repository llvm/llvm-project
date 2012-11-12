// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

typedef __attribute__(( ext_vector_type(4) )) float float4;
typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(4) )) unsigned int uint4;

// CHECK: @foo = global <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
float4 foo = (float4){ 1.0, 2.0, 3.0, 4.0 };

// CHECK: @bar = constant <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 0x7FF0000000000000>
const float4 bar = (float4){ 1.0, 2.0, 3.0, __builtin_inff() };

// CHECK: @test1
// CHECK: fadd <4 x float>
float4 test1(float4 V) {
  return V.wzyx+V;
}

float2 vec2, vec2_2;
float4 vec4, vec4_2;
float f;

// CHECK: @test2
// CHECK: shufflevector {{.*}} <i32 0, i32 1>
// CHECK: extractelement
// CHECK: shufflevector {{.*}} <i32 1, i32 1, i32 1, i32 1>
// CHECK: insertelement
// CHECK: shufflevector {{.*}} <i32 1, i32 0>
void test2() {
    vec2 = vec4.xy;  // shorten
    f = vec2.x;      // extract elt
    vec4 = vec4.yyyy;  // splat
    
    vec2.x = f;      // insert one.
    vec2.yx = vec2; // reverse
}

// CHECK: @test3
// CHECK: store <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
void test3(float4 *out) {
  *out = ((float4) {1.0f, 2.0f, 3.0f, 4.0f });
}

// CHECK: @test4
// CHECK: store <4 x float>
// CHECK: store <4 x float>
void test4(float4 *out) {
  float a = 1.0f;
  float b = 2.0f;
  float c = 3.0f;
  float d = 4.0f;
  *out = ((float4) {a,b,c,d});
}

// CHECK: @test5
// CHECK: shufflevector {{.*}} <4 x i32> zeroinitializer
// CHECK: fmul <4 x float>
// CHECK: fmul <4 x float>
// CHECK: shufflevector {{.*}} <4 x i32> zeroinitializer
// CHECK: fmul <4 x float>
void test5(float4 *out) {
  float a;
  float4 b;
  
  a = 1.0f;
  b = a;
  b = b * 5.0f;
  b = 5.0f * b;
  b *= a;
  
  *out = b;
}

// CHECK: @test6
void test6(float4 *ap, float4 *bp, float c) {
  float4 a = *ap;
  float4 b = *bp;

  // CHECK: fadd <4 x float>
  // CHECK: fsub <4 x float>
  // CHECK: fmul <4 x float>
  // CHECK: fdiv <4 x float>
  a = a + b;
  a = a - b;
  a = a * b;
  a = a / b;

  // CHECK: fadd <4 x float>
  // CHECK: fsub <4 x float>
  // CHECK: fmul <4 x float>
  // CHECK: fdiv <4 x float>
  a = a + c;
  a = a - c;
  a = a * c;
  a = a / c;

  // CHECK: fadd <4 x float>
  // CHECK: fsub <4 x float>
  // CHECK: fmul <4 x float>
  // CHECK: fdiv <4 x float>
  a += b;
  a -= b;
  a *= b;
  a /= b;

  // CHECK: fadd <4 x float>
  // CHECK: fsub <4 x float>
  // CHECK: fmul <4 x float>
  // CHECK: fdiv <4 x float>
  a += c;
  a -= c;
  a *= c;
  a /= c;

  // Vector comparisons can sometimes crash the x86 backend: rdar://6326239,
  // reject them until the implementation is stable.
#if 0
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a < b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
#endif
}

// CHECK: @test7
void test7(int4 *ap, int4 *bp, int c) {
  int4 a = *ap;
  int4 b = *bp;

  // CHECK: add <4 x i32>
  // CHECK: sub <4 x i32>
  // CHECK: mul <4 x i32>
  // CHECK: sdiv <4 x i32>
  // CHECK: srem <4 x i32>
  a = a + b;
  a = a - b;
  a = a * b;
  a = a / b;
  a = a % b;

  // CHECK: add <4 x i32>
  // CHECK: sub <4 x i32>
  // CHECK: mul <4 x i32>
  // CHECK: sdiv <4 x i32>
  // CHECK: srem <4 x i32>
  a = a + c;
  a = a - c;
  a = a * c;
  a = a / c;
  a = a % c;

  // CHECK: add <4 x i32>
  // CHECK: sub <4 x i32>
  // CHECK: mul <4 x i32>
  // CHECK: sdiv <4 x i32>
  // CHECK: srem <4 x i32>
  a += b;
  a -= b;
  a *= b;
  a /= b;
  a %= b;

  // CHECK: add <4 x i32>
  // CHECK: sub <4 x i32>
  // CHECK: mul <4 x i32>
  // CHECK: sdiv <4 x i32>
  // CHECK: srem <4 x i32>
  a += c;
  a -= c;
  a *= c;
  a /= c;
  a %= c;


  // Vector comparisons.
  // CHECK: icmp slt
  // CHECK: icmp sle
  // CHECK: icmp sgt
  // CHECK: icmp sge
  // CHECK: icmp eq
  // CHECK: icmp ne
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a > b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
}

// CHECK: @test8
void test8(float4 *ap, float4 *bp, int c) {
  float4 a = *ap;
  float4 b = *bp;

  // Vector comparisons.
  // CHECK: fcmp olt
  // CHECK: fcmp ole
  // CHECK: fcmp ogt
  // CHECK: fcmp oge
  // CHECK: fcmp oeq
  // CHECK: fcmp une
  int4 cmp;
  cmp = a < b;
  cmp = a <= b;
  cmp = a > b;
  cmp = a >= b;
  cmp = a == b;
  cmp = a != b;
}

// CHECK: @test9
// CHECK: extractelement <4 x i32>
int test9(int4 V) {
  return V.xy.x;
}

// CHECK: @test10
// CHECK: add <4 x i32>
// CHECK: extractelement <4 x i32>
int test10(int4 V) {
  return (V+V).x;
}

// CHECK: @test11
// CHECK: extractelement <4 x i32>
int4 test11a();
int test11() {
  return test11a().x;
}

// CHECK: @test12
// CHECK: shufflevector {{.*}} <i32 2, i32 1, i32 0>
// CHECK: shufflevector {{.*}} <i32 0, i32 1, i32 2, i32 undef>
// CHECK: shufflevector {{.*}} <i32 4, i32 5, i32 6, i32 3>
int4 test12(int4 V) {
  V.xyz = V.zyx;
  return V;
}

// CHECK: @test13
// CHECK: shufflevector {{.*}} <i32 2, i32 1, i32 0, i32 3>
int4 test13(int4 *V) {
  return V->zyxw;
}

// CHECK: @test14
void test14(uint4 *ap, uint4 *bp, unsigned c) {
  uint4 a = *ap;
  uint4 b = *bp;
  int4 d;
  
  // CHECK: udiv <4 x i32>
  // CHECK: urem <4 x i32>
  a = a / b;
  a = a % b;

  // CHECK: udiv <4 x i32>
  // CHECK: urem <4 x i32>
  a = a / c;
  a = a % c;

  // CHECK: icmp ult
  // CHECK: icmp ule
  // CHECK: icmp ugt
  // CHECK: icmp uge
  // CHECK: icmp eq
  // CHECK: icmp ne
  d = a < b;
  d = a <= b;
  d = a > b;
  d = a >= b;
  d = a == b;
  d = a != b;
}

// CHECK: @test15
int4 test15(uint4 V0) {
  // CHECK: icmp eq <4 x i32>
  int4 V = !V0;
  V = V && V;
  V = V || V;
  return V;
}
