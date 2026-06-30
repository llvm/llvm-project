// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL

Texture2DArray<float4> t;

// CHECK: define internal {{.*}} <4 x float> @test_mips(float vector[3])(<3 x float> {{.*}} %loc)
// CHECK: entry:
// CHECK: %[[LOC_ADDR:.*]] = alloca <3 x float>
// CHECK: %[[REF_TMP:.*]] = alloca %"struct.hlsl::Texture2DArray<>::mips_slice_type"
// CHECK: store <3 x float> %loc, ptr %[[LOC_ADDR]]
// CHECK: call void @hlsl::Texture2DArray<float vector[4]>::mips_type::operator[](int) const(ptr {{.*}} %[[REF_TMP]], ptr {{.*}} getelementptr {{.*}} (i8, ptr @t, i32 4), i32 noundef 0)
// CHECK: %[[V0:.*]] = load <3 x float>, ptr %[[LOC_ADDR]]
// CHECK: %[[SHUF:.*]] = shufflevector <3 x float> %[[V0]], <3 x float> poison, <2 x i32> <i32 0, i32 1>
// CHECK: %[[CONV:.*]] = fptosi <2 x float> %[[SHUF]] to <2 x i32>
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::mips_slice_type::operator[](int vector[3]) const(ptr {{.*}} %[[REF_TMP]], <3 x i32> {{.*}} %[[VEC:.*]])
// CHECK: ret <4 x float> %[[CALL]]

[shader("pixel")]
float4 test_mips(float3 loc : LOC) : SV_Target {
  return t.mips[0][int3(int2(loc.xy), 0)];
}

// CHECK: define linkonce_odr hidden void @hlsl::Texture2DArray<float vector[4]>::mips_type::operator[](int) const(ptr  {{.*}} %agg.result, ptr {{.*}} %this, i32 {{.*}} %Level)
// CHECK: entry:
// CHECK: %{{.*}} = alloca ptr
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LEVEL_ADDR:.*]] = alloca i32
// CHECK: %[[SLICE:.*]] = alloca %"struct.hlsl::Texture2DArray<>::mips_slice_type"
// CHECK: store ptr %agg.result, ptr %{{.*}}
// CHECK: store ptr %this, ptr %[[THIS_ADDR]]
// CHECK: store i32 %Level, ptr %[[LEVEL_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: call void @hlsl::Texture2DArray<float vector[4]>::mips_slice_type::mips_slice_type()(ptr {{.*}} %[[SLICE]])
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::Texture2DArray<>::mips_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 7), ptr %[[HANDLE_GEP]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr {{.*}} %"struct.hlsl::Texture2DArray<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 0
// CHECK: store target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], ptr %[[HANDLE_GEP2]]
// CHECK: %[[L_VAL:.*]] = load i32, ptr %[[LEVEL_ADDR]]
// CHECK: %[[LEVEL_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::Texture2DArray<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 1
// CHECK: store i32 %[[L_VAL]], ptr %[[LEVEL_GEP]]
// CHECK: call void @hlsl::Texture2DArray<float vector[4]>::mips_slice_type::mips_slice_type(hlsl::Texture2DArray<float vector[4]>::mips_slice_type const&)(ptr noundef nonnull align 4 dereferenceable(8) %agg.result, ptr noundef nonnull align 4 dereferenceable(8) %[[SLICE]])

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::mips_slice_type::operator[](int vector[3]) const(ptr {{.*}} %[[THIS:.*]], <3 x i32> noundef %[[COORD:.*]])
// CHECK: entry:
// CHECK: %[[COORD_ADDR:.*]] = alloca <3 x i32>
// CHECK: %[[VEC_TMP:.*]] = alloca <3 x i32>
// CHECK: store <3 x i32> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[COORD_PARAM:.*]] = load <3 x i32>, ptr %[[COORD_ADDR]]
// CHECK: store <3 x i32> %[[COORD_PARAM]], ptr %[[VEC_TMP]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::Texture2DArray<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 7), ptr %[[HANDLE_PTR]]
// CHECK: %[[COORD_VAL:.*]] = load <3 x i32>, ptr %[[VEC_TMP]]
// CHECK: %[[VECEXT:.*]] = extractelement <3 x i32> %[[COORD_VAL]], i32 0
// CHECK: %[[VECINIT:.*]] = insertelement <4 x i32> poison, i32 %[[VECEXT]], i32 0
// CHECK: %[[COORD_VAL2:.*]] = load <3 x i32>, ptr %[[VEC_TMP]]
// CHECK: %[[VECEXT2:.*]] = extractelement <3 x i32> %[[COORD_VAL2]], i32 1
// CHECK: %[[VECINIT3:.*]] = insertelement <4 x i32> %[[VECINIT]], i32 %[[VECEXT2]], i32 1
// CHECK: %[[COORD_VAL3:.*]] = load <3 x i32>, ptr %[[VEC_TMP]]
// CHECK: %[[VECEXT3:.*]] = extractelement <3 x i32> %[[COORD_VAL3]], i32 2
// CHECK: %[[VECINIT4:.*]] = insertelement <4 x i32> %[[VECINIT3]], i32 %[[VECEXT3]], i32 2
// CHECK: %[[LEVEL_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::Texture2DArray<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 1
// CHECK: %[[LEVEL_VAL:.*]] = load i32, ptr %[[LEVEL_PTR]]
// CHECK: %[[VECINIT5:.*]] = insertelement <4 x i32> %[[VECINIT4]], i32 %[[LEVEL_VAL]], i32 3
// CHECK: %[[COORD_X:.*]] = shufflevector <4 x i32> %[[VECINIT5]], <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK: %[[LOD:.*]] = extractelement <4 x i32> %[[VECINIT5]], i64 3
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], <3 x i32> %[[COORD_X]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]
