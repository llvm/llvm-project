// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -O1 -emit-llvm -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -O1 -emit-llvm -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK

// Matrices are handled as vectors with one element per matrix element, so the
// code is the same as for a vector of that size.

// The only difference between the targets is the calling convention on the
// definitions. It is captured once, in the first function, as [[FN_TYPE]].

// DXCHECK: define hidden [[FN_TYPE:]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f1x2{{.*}}(
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f1x2{{.*}}(
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
float1x2 test_frexp_f1x2(float1x2 X, out float1x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <3 x float> @{{.*}}test_frexp_f1x3{{.*}}(
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
float1x3 test_frexp_f1x3(float1x3 X, out float1x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f1x4{{.*}}(
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
float1x4 test_frexp_f1x4(float1x4 X, out float1x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f2x1{{.*}}(
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
float2x1 test_frexp_f2x1(float2x1 X, out float2x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f2x2{{.*}}(
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
float2x2 test_frexp_f2x2(float2x2 X, out float2x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <6 x float> @{{.*}}test_frexp_f2x3{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <6 x float> %X to <6 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <6 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <6 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <6 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<6 x i32> [[FRACTION]] to <6 x float>
// CHECK:      and <6 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<6 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <6 x float> %X, <6 x float> {{.*}}
// CHECK:      sitofp {{.*}}<6 x i32> {{.*}} to <6 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <6 x float> zeroinitializer, <6 x float> {{.*}}
// CHECK-NEXT: store <6 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <6 x float> [[MANT]]
float2x3 test_frexp_f2x3(float2x3 X, out float2x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <8 x float> @{{.*}}test_frexp_f2x4{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <8 x float> %X to <8 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <8 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <8 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <8 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<8 x i32> [[FRACTION]] to <8 x float>
// CHECK:      and <8 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<8 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <8 x float> %X, <8 x float> {{.*}}
// CHECK:      sitofp {{.*}}<8 x i32> {{.*}} to <8 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <8 x float> zeroinitializer, <8 x float> {{.*}}
// CHECK-NEXT: store <8 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <8 x float> [[MANT]]
float2x4 test_frexp_f2x4(float2x4 X, out float2x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <3 x float> @{{.*}}test_frexp_f3x1{{.*}}(
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
float3x1 test_frexp_f3x1(float3x1 X, out float3x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <6 x float> @{{.*}}test_frexp_f3x2{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <6 x float> %X to <6 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <6 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <6 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <6 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<6 x i32> [[FRACTION]] to <6 x float>
// CHECK:      and <6 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<6 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <6 x float> %X, <6 x float> {{.*}}
// CHECK:      sitofp {{.*}}<6 x i32> {{.*}} to <6 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <6 x float> zeroinitializer, <6 x float> {{.*}}
// CHECK-NEXT: store <6 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <6 x float> [[MANT]]
float3x2 test_frexp_f3x2(float3x2 X, out float3x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <9 x float> @{{.*}}test_frexp_f3x3{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <9 x float> %X to <9 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <9 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <9 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <9 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<9 x i32> [[FRACTION]] to <9 x float>
// CHECK:      and <9 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<9 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <9 x float> %X, <9 x float> {{.*}}
// CHECK:      sitofp {{.*}}<9 x i32> {{.*}} to <9 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <9 x float> zeroinitializer, <9 x float> {{.*}}
// CHECK-NEXT: store <9 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <9 x float> [[MANT]]
float3x3 test_frexp_f3x3(float3x3 X, out float3x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <12 x float> @{{.*}}test_frexp_f3x4{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <12 x float> %X to <12 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <12 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <12 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <12 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<12 x i32> [[FRACTION]] to <12 x float>
// CHECK:      and <12 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<12 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <12 x float> %X, <12 x float> {{.*}}
// CHECK:      sitofp {{.*}}<12 x i32> {{.*}} to <12 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <12 x float> zeroinitializer, <12 x float> {{.*}}
// CHECK-NEXT: store <12 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <12 x float> [[MANT]]
float3x4 test_frexp_f3x4(float3x4 X, out float3x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f4x1{{.*}}(
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
float4x1 test_frexp_f4x1(float4x1 X, out float4x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <8 x float> @{{.*}}test_frexp_f4x2{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <8 x float> %X to <8 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <8 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <8 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <8 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<8 x i32> [[FRACTION]] to <8 x float>
// CHECK:      and <8 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<8 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <8 x float> %X, <8 x float> {{.*}}
// CHECK:      sitofp {{.*}}<8 x i32> {{.*}} to <8 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <8 x float> zeroinitializer, <8 x float> {{.*}}
// CHECK-NEXT: store <8 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <8 x float> [[MANT]]
float4x2 test_frexp_f4x2(float4x2 X, out float4x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <12 x float> @{{.*}}test_frexp_f4x3{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <12 x float> %X to <12 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <12 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <12 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <12 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<12 x i32> [[FRACTION]] to <12 x float>
// CHECK:      and <12 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<12 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <12 x float> %X, <12 x float> {{.*}}
// CHECK:      sitofp {{.*}}<12 x i32> {{.*}} to <12 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <12 x float> zeroinitializer, <12 x float> {{.*}}
// CHECK-NEXT: store <12 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <12 x float> [[MANT]]
float4x3 test_frexp_f4x3(float4x3 X, out float4x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <16 x float> @{{.*}}test_frexp_f4x4{{.*}}(
// CHECK:      [[BITS:%.*]] = bitcast <16 x float> %X to <16 x i32>
// CHECK:      [[EXPBITS:%.*]] = and <16 x i32> [[BITS]], splat (i32 2139095040)
// CHECK:      [[FRACTION:%.*]] = and <16 x i32> [[BITS]], splat (i32 8388607)
// CHECK:      shl <16 x i32> [[BITS]], splat (i32 1)
// CHECK:      uitofp {{.*}}<16 x i32> [[FRACTION]] to <16 x float>
// CHECK:      and <16 x i32> [[BITS]], splat (i32 -2147483648)
// CHECK:      or {{.*}}<16 x i32> {{.*}}, splat (i32 1056964608)
// CHECK:      [[MANT:%.*]] = select {{.*}}, <16 x float> %X, <16 x float> {{.*}}
// CHECK:      sitofp {{.*}}<16 x i32> {{.*}} to <16 x float>
// CHECK:      [[EXP:%.*]] = select {{.*}}, <16 x float> zeroinitializer, <16 x float> {{.*}}
// CHECK-NEXT: store <16 x float> [[EXP]], ptr %E
// CHECK-NEXT: ret <16 x float> [[MANT]]
float4x4 test_frexp_f4x4(float4x4 X, out float4x4 E) { return frexp(X, E); }
