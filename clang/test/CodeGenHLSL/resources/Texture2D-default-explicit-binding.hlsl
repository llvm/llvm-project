// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SPIRV

SamplerState g_s : register(s0);
Texture2D<> default_template : register(t1, space2);
Texture2D implicit_template : register(t0, space1);

// CHECK: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2), %"struct.hlsl::Texture2D<>::mips_type" }
// SPIRV: %"class.hlsl::Texture2D" = type { target("spirv.Image", float, 1, 2, 0, 0, 1, 0), %"struct.hlsl::Texture2D<>::mips_type" }

// CHECK: @{{.*}}default_template = internal global %"class.hlsl::Texture2D" poison, align {{[0-9]+}}
// CHECK: @{{.*}}implicit_template = internal global %"class.hlsl::Texture2D" poison, align {{[0-9]+}}
// SPIRV: @{{.*}}default_template = internal global %"class.hlsl::Texture2D" poison, align {{[0-9]+}}
// SPIRV: @{{.*}}implicit_template = internal global %"class.hlsl::Texture2D" poison, align {{[0-9]+}}

[shader("pixel")]
float4 main(float2 uv : TEXCOORD) : SV_Target {
  return implicit_template.Sample(g_s, uv) + default_template.Sample(g_s, uv);
}

// CHECK: call void @{{.*}}__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}default_template, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// CHECK: call void @{{.*}}__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}implicit_template, i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// SPIRV: call void @{{.*}}__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}default_template, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// SPIRV: call void @{{.*}}__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}implicit_template, i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// CHECK: define void @main()
// SPIRV: define void @main()

