// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s

// CHECK-LABEL: test_lerp_half
// CHECK: [[SUB:%.*]] = fsub fast half %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast half %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast half [[MUL]], %{{.*}}
// CHECK-NEXT: ret half [[ADD]]
half test_lerp_half(half p0, half p1, half p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_half2
// CHECK: [[SUB:%.*]] = fsub fast <2 x half> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <2 x half> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <2 x half> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <2 x half> [[ADD]]
half2 test_lerp_half2(half2 p0, half2 p1, half2 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_half3
// CHECK: [[SUB:%.*]] = fsub fast <3 x half> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <3 x half> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <3 x half> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <3 x half> [[ADD]]
half3 test_lerp_half3(half3 p0, half3 p1, half3 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_half4
// CHECK: [[SUB:%.*]] = fsub fast <4 x half> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <4 x half> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <4 x half> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <4 x half> [[ADD]]
half4 test_lerp_half4(half4 p0, half4 p1, half4 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float
// CHECK: [[SUB:%.*]] = fsub fast float %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast float %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast float [[MUL]], %{{.*}}
// CHECK-NEXT: ret float [[ADD]]
float test_lerp_float(float p0, float p1, float p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float2
// CHECK: [[SUB:%.*]] = fsub fast <2 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <2 x float> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <2 x float> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <2 x float> [[ADD]]
float2 test_lerp_float2(float2 p0, float2 p1, float2 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float3
// CHECK: [[SUB:%.*]] = fsub fast <3 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <3 x float> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <3 x float> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <3 x float> [[ADD]]
float3 test_lerp_float3(float3 p0, float3 p1, float3 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float4
// CHECK: [[SUB:%.*]] = fsub fast <4 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <4 x float> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <4 x float> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <4 x float> [[ADD]]
float4 test_lerp_float4(float4 p0, float4 p1, float4 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float5
// CHECK: [[SUB:%.*]] = fsub fast <5 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT: [[MUL:%.*]] = fmul fast <5 x float> %{{.*}}, [[SUB]]
// CHECK-NEXT: [[ADD:%.*]] = fadd fast <5 x float> [[MUL]], %{{.*}}
// CHECK-NEXT: ret <5 x float> [[ADD]]
vector<float, 5> test_lerp_float5(vector<float, 5> p0,
							      vector<float, 5> p1,
							      vector<float, 5> p2) {
	return lerp(p0, p1, p2);
}
