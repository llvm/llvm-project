// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,STABLE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -o - %s | FileCheck %s --check-prefixes=CHECK,STABLE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -o - %s | FileCheck %s --check-prefixes=CHECK,OPTIMIZED
// RUN: %clang_dxc -T lib_6_3 -fcgl %s | FileCheck %s --check-prefixes=CHECK,STABLE
// RUN: %clang_dxc -T lib_6_3 -fcgl -pack-prefix-stable %s | FileCheck %s --check-prefixes=CHECK,STABLE
// RUN: %clang_dxc -T lib_6_3 -fcgl -pack-optimized %s | FileCheck %s --check-prefixes=CHECK,OPTIMIZED

struct Varyings {
  float a : A;
  float2 b : B;
  float3 c : C;
  float2 d : D;
};

struct Attributes {
  float a : IA;
  float2 b : IB;
  float3 c : IC;
  float2 d : ID;
};

[shader("vertex")]
Varyings vs_main(Attributes input) {
  Varyings output = {input.a, input.b, input.c, input.d};
  return output;
}

struct Targets {
  float4 color : SV_Target3;
  float extra : SV_Target0;
};

[shader("pixel")]
Targets ps_main(Varyings input) {
  Targets output;
  output.color = float4(input.a, input.b, input.c.x);
  output.extra = input.d.x;
  return output;
}

// Signature IDs and intrinsic operands stay in declaration order even when
// optimized packing assigns locations in a different order.
// CHECK: call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0,
// CHECK: call void @llvm.dx.store.output.v2f32(i32 1, i32 0, i8 0,
// CHECK: call void @llvm.dx.store.output.v3f32(i32 2, i32 0, i8 0,
// CHECK: call void @llvm.dx.store.output.v2f32(i32 3, i32 0, i8 0,
// CHECK: call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0,
// CHECK: call <2 x float> @llvm.dx.load.input.v2f32(i32 1, i32 0, i8 0,
// CHECK: call <3 x float> @llvm.dx.load.input.v3f32(i32 2, i32 0, i8 0,
// CHECK: call <2 x float> @llvm.dx.load.input.v2f32(i32 3, i32 0, i8 0,

// Use each entry's shader attribute, rather than the library target stage.
// Vertex outputs and pixel inputs have identical packed metadata.
// CHECK: !dx.semantic.signatures = !{![[VS:[0-9]+]], ![[PS:[0-9]+]]}
// CHECK-DAG: ![[VS]] = !{ptr @vs_main, ![[VSIN:[0-9]+]], ![[VARY:[0-9]+]]}
// CHECK-DAG: ![[PS]] = !{ptr @ps_main, ![[VARY]], ![[PSOUT:[0-9]+]]}
// CHECK-DAG: ![[VSIN]] = !{![[IA:[0-9]+]], ![[IB:[0-9]+]], ![[IC:[0-9]+]], ![[ID:[0-9]+]]}
// CHECK-DAG: ![[VARY]] = !{![[A:[0-9]+]], ![[B:[0-9]+]], ![[C:[0-9]+]], ![[D:[0-9]+]]}
// CHECK-DAG: ![[PSOUT]] = !{![[TARGET3:[0-9]+]], ![[TARGET0:[0-9]+]]}

// Vertex inputs are stacked in declaration order regardless of packing mode.
// CHECK-DAG: ![[IA]] = !{i32 0, !"IA", i32 9, i32 0, ![[ZERO:[0-9]+]], i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[IB]] = !{i32 1, !"IB", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 1, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[IC]] = !{i32 2, !"IC", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 3, i32 2, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[ID]] = !{i32 3, !"ID", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 3, i8 0, i8 0, i8 0, i32 0}

// Prefix-stable packing uses three rows: (a,b), (c), (d).
// STABLE-DAG: ![[A]] = !{i32 0, !"A", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// STABLE-DAG: ![[B]] = !{i32 1, !"B", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 0, i8 1, i8 0, i8 0, i32 0}
// STABLE-DAG: ![[C]] = !{i32 2, !"C", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 3, i32 1, i8 0, i8 0, i8 0, i32 0}
// STABLE-DAG: ![[D]] = !{i32 3, !"D", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 2, i8 0, i8 0, i8 0, i32 0}

// Optimized packing uses two rows: (c,a), (b,d).
// OPTIMIZED-DAG: ![[A]] = !{i32 0, !"A", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 1, i32 0, i8 3, i8 0, i8 0, i32 0}
// OPTIMIZED-DAG: ![[B]] = !{i32 1, !"B", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 1, i8 0, i8 0, i8 0, i32 0}
// OPTIMIZED-DAG: ![[C]] = !{i32 2, !"C", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 3, i32 0, i8 0, i8 0, i8 0, i32 0}
// OPTIMIZED-DAG: ![[D]] = !{i32 3, !"D", i32 9, i32 0, ![[ZERO]], i32 0, i32 1, i8 2, i32 1, i8 2, i8 0, i8 0, i32 0}

// Pixel output rows always follow semantic indices, including gaps, not
// declaration order or the selected packing mode.
// CHECK-DAG: ![[TARGET3]] = !{i32 0, !"SV_Target", i32 9, i32 {{[0-9]+}}, ![[THREE:[0-9]+]], i32 0, i32 1, i8 4, i32 3, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[TARGET0]] = !{i32 1, !"SV_Target", i32 9, i32 {{[0-9]+}}, ![[ZERO]], i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DAG: ![[ZERO]] = !{i32 0}
// CHECK-DAG: ![[THREE]] = !{i32 3}
