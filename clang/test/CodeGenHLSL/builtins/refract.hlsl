// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=SPVCHECK

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) half @_Z17test_refract_halfDhDhDh(
// CHECK-SAME: half noundef nofpclass(nan inf) [[I:%.*]], half noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL1_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL2_I]]
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL1_I]], [[SUB_I]]
// CHECK:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL3_I]]
// CHECK:    [[MUL5_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[TMP0:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(half %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn half [[MUL6_I]], %{{.*}}
// CHECK:    [[MUL7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ADD_I]], %{{.*}}
// CHECK:    [[SUB8_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half %{{.*}}, [[MUL7_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half %{{.*}}, 0xH0000
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CMP_I]], half 0xH0000, half %{{.*}}
// CHECK:    ret half [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) half @_Z17test_refract_halfDhDhDh(
// SPVCHECK-SAME: half noundef nofpclass(nan inf) [[I:%.*]], half noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] 
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef half @llvm.spv.refract.f16.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// SPVCHECK:    ret half [[SPV_REFRACT_I]]
//
half test_refract_half(half I, half N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x half> @_Z18test_refract_half2Dv2_DhS_Dh(
// CHECK-SAME: <2 x half> noundef nofpclass(nan inf) [[I:%.*]], <2 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x half> splat (half 0xH3C00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x half> splat (half 0xH3C00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.sqrt.v2f16(<2 x half> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x half> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x half> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <2 x half> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <2 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <2 x half> zeroinitializer, <2 x half> %{{.*}}
// CHECK:    ret <2 x half> [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <2 x half> @_Z18test_refract_half2Dv2_DhS_Dh(
// SPVCHECK-SAME: <2 x half> noundef nofpclass(nan inf) [[I:%.*]], <2 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <2 x half> @llvm.spv.refract.v2f16.f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, half %{{.*}})
// SPVCHECK:    ret <2 x half> [[SPV_REFRACT_I]]
//
half2 test_refract_half2(half2 I, half2 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x half> @_Z18test_refract_half3Dv3_DhS_Dh(
// CHECK-SAME: <3 x half> noundef nofpclass(nan inf) [[I:%.*]], <3 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x half> splat (half 0xH3C00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x half> splat (half 0xH3C00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.sqrt.v3f16(<3 x half> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x half> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x half> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <3 x half> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <3 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <3 x half> zeroinitializer, <3 x half> %{{.*}}
// CHECK:    ret <3 x half> [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <3 x half> @_Z18test_refract_half3Dv3_DhS_Dh(
// SPVCHECK-SAME: <3 x half> noundef nofpclass(nan inf) [[I:%.*]], <3 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <3 x half> @llvm.spv.refract.v3f16.f16(<3 x half> %{{.*}}, <3 x half> %{{.*}}, half %{{.*}})
// SPVCHECK:    ret <3 x half> [[SPV_REFRACT_I]]
//
half3 test_refract_half3(half3 I, half3 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x half> @_Z18test_refract_half4Dv4_DhS_Dh(
// CHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[I:%.*]], <4 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x half> splat (half 0xH3C00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x half> splat (half 0xH3C00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.sqrt.v4f16(<4 x half> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x half> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x half> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <4 x half> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <4 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <4 x half> zeroinitializer, <4 x half> %{{.*}}
// CHECK:    ret <4 x half> [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <4 x half> @_Z18test_refract_half4Dv4_DhS_Dh(
// SPVCHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[I:%.*]], <4 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x half> @llvm.spv.refract.v4f16.f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, half %{{.*}})
// SPVCHECK:    ret <4 x half> [[SPV_REFRACT_I]]
//
half4 test_refract_half4(half4 I, half4 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float @_Z18test_refract_floatfff(
// CHECK-SAME: float noundef nofpclass(nan inf) [[I:%.*]], float noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL1_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL2_I]]
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL1_I]], [[SUB_I]]
// CHECK:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL3_I]]
// CHECK:    [[MUL5_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(float %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn float [[MUL6_I]], %{{.*}}
// CHECK:    [[MUL7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ADD_I]], %{{.*}}
// CHECK:    [[SUB8_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL7_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CMP_I]], float 0.000000e+00, float %{{.*}}
// CHECK:    ret float [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) float @_Z18test_refract_floatfff(
// SPVCHECK-SAME: float noundef nofpclass(nan inf) [[I:%.*]], float noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef float @llvm.spv.refract.f32.f32(float %{{.*}}, float  %{{.*}}, float %{{.*}})
// SPVCHECK:    ret float [[SPV_REFRACT_I]]
//
float test_refract_float(float I, float N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z19test_refract_float2Dv2_fS_f(
// CHECK-SAME: <2 x float> noundef nofpclass(nan inf) [[I:%.*]], <2 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <2 x float> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <2 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <2 x float> zeroinitializer, <2 x float> %{{.*}}
// CHECK:    ret <2 x float> [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <2 x float> @_Z19test_refract_float2Dv2_fS_f(
// SPVCHECK-SAME: <2 x float> noundef nofpclass(nan inf) [[I:%.*]], <2 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <2 x float> @llvm.spv.refract.v2f32.f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, float %{{.*}})
// SPVCHECK:    ret <2 x float> [[SPV_REFRACT_I]]
//
float2 test_refract_float2(float2 I, float2 N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> @_Z19test_refract_float3Dv3_fS_f(
// CHECK-SAME: <3 x float> noundef nofpclass(nan inf) [[I:%.*]], <3 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32(<3 x float> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <3 x float> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <3 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <3 x float> zeroinitializer, <3 x float> %{{.*}}
// CHECK:    ret <3 x float> [[HLSL_SELECT_I]]
//
// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <3 x float> @_Z19test_refract_float3Dv3_fS_f(
// SPVCHECK-SAME: <3 x float> noundef nofpclass(nan inf) [[I:%.*]], <3 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <3 x float> @llvm.spv.refract.v3f32.f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, float %{{.*}})
// SPVCHECK:    ret <3 x float> [[SPV_REFRACT_I]]
//
float3 test_refract_float3(float3 I, float3 N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z19test_refract_float4Dv4_fS_f
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[I:%.*]], <4 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[HLSL_DOT_I:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), [[MUL3_I]]
// CHECK:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB_I]]
// CHECK:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), [[MUL4_I]]
// CHECK:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL11_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[TMP17:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.*}})
// CHECK:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> [[MUL11_I]], [[TMP17]]
// CHECK:    [[MUL12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> [[ADD_I]], %{{.*}}
// CHECK:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> [[MUL8_I]], [[MUL12_I]]
// CHECK:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt  <4 x float> %{{.*}}, zeroinitializer
// CHECK:    [[CAST:%.*]] = extractelement <4 x i1> [[CMP_I]], i32 0
// CHECK:    [[HLSL_SELECT_I:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CAST]], <4 x float> zeroinitializer, <4 x float> %{{.*}}
// CHECK:    ret <4 x float> [[HLSL_SELECT_I]]

// SPVCHECK-LABEL: define hidden spir_func noundef nofpclass(nan inf) <4 x float> @_Z19test_refract_float4Dv4_fS_f(
// SPVCHECK-SAME: <4 x float> noundef nofpclass(nan inf) %{{.*}}, <4 x float> noundef nofpclass(nan inf) %{{.*}}, float noundef nofpclass(nan inf) %{{.*}}) #[[ATTR0:[0-9]+]] {
// SPVCHECK:  [[ENTRY:.*:]]
// SPVCHECK:    [[SPV_REFRACT_I:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.spv.refract.v4f32.f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, float %{{.*}})
// SPVCHECK:    ret <4 x float> [[SPV_REFRACT_I]]
//
float4 test_refract_float4(float4 I, float4 N, float ETA) {
    return refract(I, N, ETA);
}
