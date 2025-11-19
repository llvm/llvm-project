// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: test_lit_half
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %{{.*}}, 0xH0000
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, half 0xH0000, half %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x half> <half 0xH3C00, half poison, half poison, half poison>, half %{{.*}}, i32 1
// CHECK: %vecinit2.i = insertelement <4 x half> %{{.*}}, half 0xH3C00, i32 3
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt half %{{.*}}, 0xH0000
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: [[LOG:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.log.f16(half %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn half [[LOG]], %{{.*}}
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.exp.f16(half %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, half 0xH0000, half %{{.*}}
// CHECK: %vecins.i = insertelement <4 x half> %{{.*}}, half %hlsl.select7.i, i32 2
// CHECK: ret <4 x half> %{{.*}}
half4 test_lit_half(half NDotL, half NDotH, half M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %vecinit2.i = insertelement <4 x float> %{{.*}}, float 1.000000e+00, i32 3
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: [[LOG:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float [[LOG]], %{{.*}}
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: ret <4 x float> %{{.*}}
float4 test_lit_float(float NDotL, float NDotH, float M) { return lit(NDotL, NDotH, M); }
