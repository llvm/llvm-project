// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -O1 -emit-llvm -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -O1 -emit-llvm -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK

// The code works on the bits of the float, so it does not depend on how
// denormals are handled. A denormal is normalized by converting its fraction
// to a float, which is why there is a uitofp. The sign is kept by or-ing in the
// sign bit, and the exponent field of 0.5 is or-ed in, which is 0x3f000000.

// The only difference between the targets is the calling convention on the
// definitions. It is captured once, in the first function, as [[FN_TYPE]].

// DXCHECK: define hidden [[FN_TYPE:]]noundef nofpclass(nan inf) float @{{.*}}test_frexp_f{{.*}}(
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef nofpclass(nan inf) float @{{.*}}test_frexp_f{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast float %X to i32
// CHECK:      [[EXPBITS:%.*]] = and i32 [[BITS]], 2139095040
// CHECK:      [[FRACTION:%.*]] = and i32 [[BITS]], 8388607
// CHECK:      shl i32 [[BITS]], 1
// CHECK:      uitofp {{.*}}i32 [[FRACTION]] to float
// CHECK:      and i32 [[BITS]], -2147483648
// CHECK:      or {{.*}}i32 {{.*}}, 1056964608
// CHECK:      [[MANT:%.*]] = select {{.*}}, float %X, float {{.*}}
// CHECK:      sitofp {{.*}}i32 {{.*}} to float
// CHECK:      [[EXP:%.*]] = select {{.*}}, float 0.000000e+00, float {{.*}}
// CHECK-NEXT: store float [[EXP]], ptr %E
// CHECK-NEXT: ret float [[MANT]]
float test_frexp_f(float X, out float E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f2{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <2 x float> %X to <2 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <2 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <2 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <2 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<2 x i32> [[FRACTION]] to <2 x float>
// CHECK:      and <2 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<2 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <2 x float> %X, <2 x float> {{.*}}
// CHECK:      sitofp {{.*}}<2 x i32> {{.*}} to <2 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <2 x float> zeroinitializer, <2 x float> {{.*}}
// CHECK-NEXT: store <2 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <2 x float> [[MANT]]
float2 test_frexp_f2(float2 X, out float2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <3 x float> @{{.*}}test_frexp_f3{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <3 x float> %X to <3 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <3 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <3 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <3 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<3 x i32> [[FRACTION]] to <3 x float>
// CHECK:      and <3 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<3 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <3 x float> %X, <3 x float> {{.*}}
// CHECK:      sitofp {{.*}}<3 x i32> {{.*}} to <3 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <3 x float> zeroinitializer, <3 x float> {{.*}}
// CHECK-NEXT: store <3 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <3 x float> [[MANT]]
float3 test_frexp_f3(float3 X, out float3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f4{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <4 x float> %X to <4 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <4 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <4 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <4 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<4 x i32> [[FRACTION]] to <4 x float>
// CHECK:      and <4 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<4 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <4 x float> %X, <4 x float> {{.*}}
// CHECK:      sitofp {{.*}}<4 x i32> {{.*}} to <4 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <4 x float> zeroinitializer, <4 x float> {{.*}}
// CHECK-NEXT: store <4 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <4 x float> [[MANT]]
float4 test_frexp_f4(float4 X, out float4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <5 x float> @{{.*}}test_frexp_f5{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <5 x float> %X to <5 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <5 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <5 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <5 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<5 x i32> [[FRACTION]] to <5 x float>
// CHECK:      and <5 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<5 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <5 x float> %X, <5 x float> {{.*}}
// CHECK:      sitofp {{.*}}<5 x i32> {{.*}} to <5 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <5 x float> zeroinitializer, <5 x float> {{.*}}
// CHECK-NEXT: store <5 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <5 x float> [[MANT]]
vector<float, 5> test_frexp_f5(vector<float, 5> X, out vector<float, 5> E) {
  return frexp(X, E);
}

// The exponent can also be an integer, as with DXC. It is computed as a float
// and converted when it is written back, which folds away here because the
// exponent is an exact integer.

// CHECK: define {{.*}}test_frexp_int_exp_scalar
// CHECK:      [[EXPI:%.*]] = select {{.*}}, i32 0, i32 {{.*}}
// CHECK-NEXT: store i32 [[EXPI]], ptr %E
// CHECK-NEXT: ret float
float test_frexp_int_exp_scalar(float X, out int E) { return frexp(X, E); }

// CHECK: define {{.*}}test_frexp_int_exp_long
// CHECK:      [[EXPI:%.*]] = select {{.*}}, <8 x i32> zeroinitializer, <8 x i32> {{.*}}
// CHECK-NEXT: store <8 x i32> [[EXPI]], ptr %E
// CHECK-NEXT: ret <8 x float>
vector<float, 8> test_frexp_int_exp_long(vector<float, 8> X,
                                         out vector<int, 8> E) {
  return frexp(X, E);
}
