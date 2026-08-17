// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips -DTEXTURE=Texture2D -DCOORD_DIM=2 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,NOARRAY -DTEXTURE=Texture2D -DCOORD_DIM=2 -DLOCATION_DIM=3 -DDXIL_TY=2
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,ARRAY -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -DLOCATION_DIM=4 -DDXIL_TY=7

TEXTURE<float4> t;

// CHECK: define internal {{.*}} <4 x float> @test_mips(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> {{.*}} %loc)
// CHECK: entry:
// CHECK: %[[LOC_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[REF_TMP:.*]] = alloca %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type"
// CHECK: store <[[COORD_DIM]] x float> %loc, ptr %[[LOC_ADDR]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_type::operator[](int) const(ptr {{.*}} %[[REF_TMP]], ptr {{.*}} getelementptr {{.*}} (i8, ptr @t, i32 4), i32 noundef 0)
// CHECK: %[[V0:.*]] = load <[[COORD_DIM]] x float>, ptr %[[LOC_ADDR]]
// CHECK: %[[CONV:.*]] = fptosi <[[COORD_DIM]] x float> %[[V0]] to <[[COORD_DIM]] x i32>
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::operator[](int vector[[[COORD_DIM]]]) const(ptr {{.*}} %[[REF_TMP]], <[[COORD_DIM]] x i32> {{.*}} %[[CONV]])
// CHECK: ret <4 x float> %[[CALL]]

[shader("pixel")]
float4 test_mips(vector<float, COORD_DIM> loc : LOC) : SV_Target {
  return t.mips[0][(vector<int, COORD_DIM>)loc];
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::mips_type::operator[](int) const(ptr  {{.*}} %agg.result, ptr {{.*}} %this, i32 {{.*}} %Level)
// CHECK: entry:
// CHECK: %{{.*}} = alloca ptr
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LEVEL_ADDR:.*]] = alloca i32
// CHECK: %[[SLICE:.*]] = alloca %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type"
// CHECK: store ptr %agg.result, ptr %{{.*}}
// CHECK: store ptr %this, ptr %[[THIS_ADDR]]
// CHECK: store i32 %Level, ptr %[[LEVEL_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::mips_slice_type()(ptr {{.*}} %[[SLICE]])
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 0
// CHECK: store target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[HANDLE]], ptr %[[HANDLE_GEP2]]
// CHECK: %[[L_VAL:.*]] = load i32, ptr %[[LEVEL_ADDR]]
// CHECK: %[[LEVEL_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 1
// CHECK: store i32 %[[L_VAL]], ptr %[[LEVEL_GEP]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::mips_slice_type(hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type const&)(ptr noundef nonnull align 4 dereferenceable(8) %agg.result, ptr noundef nonnull align 4 dereferenceable(8) %[[SLICE]])

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::operator[](int vector[[[COORD_DIM]]]) const(ptr {{.*}} %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[COORD:.*]])
// CHECK: entry:
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: %[[VEC_TMP:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store <[[COORD_DIM]] x i32> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[COORD_PARAM:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[COORD_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[COORD_PARAM]], ptr %[[VEC_TMP]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[VEC_TMP]]
// CHECK: %[[VECEXT:.*]] = extractelement <[[COORD_DIM]] x i32> %[[COORD_VAL]], i32 0
// CHECK: %[[VECINIT:.*]] = insertelement <[[LOCATION_DIM]] x i32> poison, i32 %[[VECEXT]], i32 0
// CHECK: %[[COORD_VAL2:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[VEC_TMP]]
// CHECK: %[[VECEXT2:.*]] = extractelement <[[COORD_DIM]] x i32> %[[COORD_VAL2]], i32 1
// CHECK: %[[VECINIT3:.*]] = insertelement <[[LOCATION_DIM]] x i32> %[[VECINIT]], i32 %[[VECEXT2]], i32 1
// ARRAY: %[[COORD_VAL3:.*]] = load <3 x i32>, ptr %[[VEC_TMP]]
// ARRAY: %[[VECEXT3:.*]] = extractelement <3 x i32> %[[COORD_VAL3]], i32 2
// ARRAY: %[[VECINIT3B:.*]] = insertelement <4 x i32> %[[VECINIT3]], i32 %[[VECEXT3]], i32 2
// CHECK: %[[LEVEL_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 1
// CHECK: %[[LEVEL_VAL:.*]] = load i32, ptr %[[LEVEL_PTR]]
// NOARRAY: %[[VECINITL:.*]] = insertelement <3 x i32> %[[VECINIT3]], i32 %[[LEVEL_VAL]], i32 2
// ARRAY: %[[VECINITL:.*]] = insertelement <4 x i32> %[[VECINIT3B]], i32 %[[LEVEL_VAL]], i32 3
// CHECK: %[[COORD_X:.*]] = shufflevector <[[LOCATION_DIM]] x i32> %[[VECINITL]], <[[LOCATION_DIM]] x i32> poison, <[[COORD_DIM]] x i32> {{.*}}
// CHECK: %[[LOD:.*]] = extractelement <[[LOCATION_DIM]] x i32> %[[VECINITL]], i64 [[COORD_DIM]]
// CHECK: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_[[DXIL_TY]]t.v[[COORD_DIM]]i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD_X]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]
