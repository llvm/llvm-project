// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN: dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN: -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK \
// RUN: -DTARGET=dx

// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN: spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN: -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK \
// RUN: -DTARGET=spv


// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) half @_Z16test_length_halfDh(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) half @_Z16test_length_halfDh(
// CHECK-SAME: half noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[ELT_ABS_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef half @llvm.fabs.f16(half [[P0]])
// CHECK-NEXT:    ret half [[ELT_ABS_I]]
//

half test_length_half(half p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) half @_Z17test_length_half2Dv2_Dh(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) half @_Z17test_length_half2Dv2_Dh(
// CHECK-SAME: <2 x half> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].fdot.v2f16(<2 x half> [[P0]], <2 x half> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef half @llvm.sqrt.f16(half [[HLSL_DOT_I]])
// CHECK-NEXT:    ret half [[TMP0]]
//


half test_length_half2(half2 p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) half @_Z17test_length_half3Dv3_Dh(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) half @_Z17test_length_half3Dv3_Dh(
// CHECK-SAME: <3 x half> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].fdot.v3f16(<3 x half> [[P0]], <3 x half> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef half @llvm.sqrt.f16(half [[HLSL_DOT_I]])
// CHECK-NEXT:    ret half [[TMP0]]
//
half test_length_half3(half3 p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) half @_Z17test_length_half4Dv4_Dh(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) half @_Z17test_length_half4Dv4_Dh(
// CHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].fdot.v4f16(<4 x half> [[P0]], <4 x half> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef half @llvm.sqrt.f16(half [[HLSL_DOT_I]])
// CHECK-NEXT:    ret half [[TMP0]]
//
half test_length_half4(half4 p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) float @_Z17test_length_floatf(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) float @_Z17test_length_floatf(
// CHECK-SAME: float noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[ELT_ABS_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef float @llvm.fabs.f32(float [[P0]])
// CHECK-NEXT:    ret float [[ELT_ABS_I]]
//
float test_length_float(float p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) float @_Z18test_length_float2Dv2_f(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) float @_Z18test_length_float2Dv2_f(
// CHECK-SAME: <2 x float> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].fdot.v2f32(<2 x float> [[P0]], <2 x float> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef float @llvm.sqrt.f32(float [[HLSL_DOT_I]])
// CHECK-NEXT:    ret float [[TMP0]]
//
float test_length_float2(float2 p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) float @_Z18test_length_float3Dv3_f(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) float @_Z18test_length_float3Dv3_f(
// CHECK-SAME: <3 x float> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].fdot.v3f32(<3 x float> [[P0]], <3 x float> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef float @llvm.sqrt.f32(float [[HLSL_DOT_I]])
// CHECK-NEXT:    ret float [[TMP0]]
//
float test_length_float3(float3 p0)
{
  return length(p0);
}

// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) float @_Z18test_length_float4Dv4_f(
// DXCHECK-LABEL: define noundef nofpclass(nan inf) float @_Z18test_length_float4Dv4_f(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[P0:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].fdot.v4f32(<4 x float> [[P0]], <4 x float> [[P0]])
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef float @llvm.sqrt.f32(float [[HLSL_DOT_I]])
// CHECK-NEXT:    ret float [[TMP0]]
//
float test_length_float4(float4 p0)
{
  return length(p0);
}
