// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: test_lit_double
// CHECK: %conv.i = fptrunc reassoc nnan ninf nsz arcp afn double %{{.*}} to float
// CHECK: %conv1.i = fptrunc reassoc nnan ninf nsz arcp afn double %{{.*}} to float
// CHECK: %conv2.i = fptrunc reassoc nnan ninf nsz arcp afn double %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %conv3.i = fpext reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}} to <4 x double>
// CHECK: ret <4 x double> %conv3.i
double4 test_lit_double(double NDotL, double NDotH, double M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_int
// CHECK: %conv.i = sitofp i32 %{{.*}} to float
// CHECK: %conv1.i = sitofp i32 %{{.*}} to float
// CHECK: %conv2.i = sitofp i32 %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %conv3.i = fptosi <4 x float> %{{.*}} to <4 x i32>
// CHECK: ret <4 x i32> %conv3.i
int4 test_lit_int(int NDotL, int NDotH, int M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_uint
// CHECK: %conv.i = uitofp i32 %{{.*}} to float
// CHECK: %conv1.i = uitofp i32 %{{.*}} to float
// CHECK: %conv2.i = uitofp i32 %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %conv3.i = fptoui <4 x float> %{{.*}} to <4 x i32>
// CHECK: ret <4 x i32> %conv3.i
uint4 test_lit_uint(uint NDotL, uint NDotH, uint M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_int64_t
// CHECK: %conv.i = sitofp i64 %{{.*}} to float
// CHECK: %conv1.i = sitofp i64 %{{.*}} to float
// CHECK: %conv2.i = sitofp i64 %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %conv3.i = fptosi <4 x float> %{{.*}} to <4 x i64>
// CHECK: ret <4 x i64> %conv3.i
int64_t4 test_lit_int64_t(int64_t NDotL, int64_t NDotH, int64_t M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_uint64_t
// CHECK: %conv.i = uitofp i64 %{{.*}} to float
// CHECK: %conv1.i = uitofp i64 %{{.*}} to float
// CHECK: %conv2.i = uitofp i64 %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %conv3.i = fptoui <4 x float> %{{.*}} to <4 x i64>
// CHECK: ret <4 x i64> %conv3.i
uint64_t4 test_lit_uint64_t(uint64_t NDotL, uint64_t NDotH, uint64_t M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: test_lit_bool
// CHECK: %conv.i = uitofp i1 %{{.*}} to float
// CHECK: %conv4.i = uitofp i1 %{{.*}} to float
// CHECK: %conv6.i = uitofp i1 %{{.*}} to float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecinit.i = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float %{{.*}}, i32 1
// CHECK: %cmp4.i = fcmp reassoc nnan ninf nsz arcp afn olt float %{{.*}}, 0.000000e+00
// CHECK: %hlsl.or.i = or i1 %{{.*}}, %cmp4.i
// CHECK: %elt.log.i = call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(float %{{.*}})
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %elt.log.i, %{{.*}}
// CHECK: %elt.exp.i = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(float %mul.i)
// CHECK: %hlsl.select7.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float 0.000000e+00, float %{{.*}}
// CHECK: %vecins.i = insertelement <4 x float> %{{.*}}, float %hlsl.select7.i, i32 2
// CHECK: %tobool.i = fcmp reassoc nnan ninf nsz arcp afn une <4 x float> %{{.*}}, zeroinitializer
// CHECK: ret <4 x i1> %tobool.i
bool4 test_lit_bool(bool NDotL, bool NDotH, bool M) { return lit(NDotL, NDotH, M); }
