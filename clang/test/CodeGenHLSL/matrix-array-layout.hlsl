// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// StructuredBuffer is read-only and stores one handle per resource.
// CHECK: type { target("dx.RawBuffer", [3 x <2 x float>], 0, 0) } 
// CHECK: type { target("dx.RawBuffer", [2 x <3 x float>], 0, 0) }

// CHECK: type { target("dx.RawBuffer", [2 x <3 x float>], 1, 0), target("dx.RawBuffer", [2 x <3 x float>], 1, 0) }
// CHECK: type { target("dx.RawBuffer", [3 x <2 x float>], 1, 1), target("dx.RawBuffer", [3 x <2 x float>], 1, 1) }

// The array element layout matches the bare matrix layout for each orientation.
// CHECK: %rm_arr = alloca [2 x [2 x <3 x float>]], align 4
// CHECK: %cm_arr = alloca [2 x [3 x <2 x float>]], align 4
// CHECK: %rm_bare = alloca [2 x <3 x float>], align 4
// CHECK: %cm_bare = alloca [3 x <2 x float>], align 4
// CHECK: %[[RM_VALUE:.*]] = load <6 x float>, ptr %rm_bare, align 4
// CHECK: %[[RM_ELEMENT:.*]] = getelementptr inbounds [2 x [2 x <3 x float>]], ptr %rm_arr, i32 0, i32 0
// CHECK: store <6 x float> %[[RM_VALUE]], ptr %[[RM_ELEMENT]], align 4
// CHECK: %[[CM_VALUE:.*]] = load <6 x float>, ptr %cm_bare, align 4
// CHECK: %[[CM_ELEMENT:.*]] = getelementptr inbounds [2 x [3 x <2 x float>]], ptr %cm_arr, i32 0, i32 0
// CHECK: store <6 x float> %[[CM_VALUE]], ptr %[[CM_ELEMENT]], align 4

export void f() {
  row_major    float2x3 rm_arr[2];
  column_major float2x3 cm_arr[2];
  row_major    float2x3 rm_bare;
  column_major float2x3 cm_bare;
  rm_arr[0] = rm_bare;
  cm_arr[0] = cm_bare;
}

float use_default_layout(float2x3 M) { return M[0][0]; }

export float call_default_layout(row_major float2x3 M) {
  return use_default_layout(M);
}

// CHECK-LABEL: define {{.*}} float @_Z19call_default_layoutu11matrix_typeILm2ELm3ELm1EfE
// CHECK: %[[CALL_LAYOUT:.*]] = shufflevector <6 x float> %{{.*}}, <6 x float> poison, <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
// CHECK: call {{.*}} float @_Z18use_default_layoutu11matrix_typeILm2ELm3EfE(<6 x float> {{.*}}%[[CALL_LAYOUT]])

StructuredBuffer<column_major float2x3> ColumnSource : register(t0);
StructuredBuffer<row_major float2x3> RowSource : register(t1);
RWStructuredBuffer<row_major float2x3> RowDestination : register(u0);
RasterizerOrderedStructuredBuffer<column_major float2x3> ColumnDestination
  : register(u1);

[numthreads(1,1,1)]
void main() {
  RowDestination[0] = ColumnSource[0];
  ColumnDestination[0] = RowDestination[0];
  RowDestination[1] = RowSource[0];
}

// CHECK-LABEL: define internal void @_Z4mainv()
// CHECK: %[[CM_PTR:.*]] = call {{.*}} ptr {{.*}}StructuredBuffer{{.*}}ColumnSource
// CHECK: %[[CM_LOAD:.*]] = load <6 x float>, ptr %[[CM_PTR]], align 4
// CHECK: %[[CM_TO_RM:.*]] = shufflevector <6 x float> %[[CM_LOAD]], <6 x float> poison, <6 x i32> <i32 0, i32 2, i32 4, i32 1, i32 3, i32 5>
// CHECK: %[[RM_PTR:.*]] = call {{.*}} ptr {{.*}}RWStructuredBuffer{{.*}}RowDestination
// CHECK: store <6 x float> %[[CM_TO_RM]], ptr %[[RM_PTR]], align 4
// CHECK: %[[RM_SRC_PTR:.*]] = call {{.*}} ptr {{.*}}RWStructuredBuffer{{.*}}RowDestination
// CHECK: %[[RM_LOAD:.*]] = load <6 x float>, ptr %[[RM_SRC_PTR]], align 4
// CHECK: %[[RM_TO_CM:.*]] = shufflevector <6 x float> %[[RM_LOAD]], <6 x float> poison, <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
// CHECK: %[[CM_DST_PTR:.*]] = call {{.*}} ptr {{.*}}RasterizerOrderedStructuredBuffer{{.*}}ColumnDestination
// CHECK: store <6 x float> %[[RM_TO_CM]], ptr %[[CM_DST_PTR]], align 4

// CHECK-LABEL: define linkonce_odr hidden {{.*}} ptr @_ZNK4hlsl16StructuredBufferIu11matrix_typeILm2ELm3ELm2EfEEixEj
// CHECK: %[[CM_HANDLE_PTR:.*]] = getelementptr {{.*}}%"class.hlsl::StructuredBuffer", ptr {{.*}}, i32 0, i32 0
// CHECK: %[[CM_HANDLE:.*]] = load target("dx.RawBuffer", [3 x <2 x float>], 0, 0), ptr %[[CM_HANDLE_PTR]], align 4
// CHECK: call ptr @llvm.dx.resource.getpointer{{.*}}(target("dx.RawBuffer", [3 x <2 x float>], 0, 0) %[[CM_HANDLE]], i32 {{.*}})

// CHECK-LABEL: define linkonce_odr hidden {{.*}} ptr @_ZNK4hlsl18RWStructuredBuffer
// CHECK: %[[RM_HANDLE_PTR:.*]] = getelementptr {{.*}}%"class.hlsl::RWStructuredBuffer", ptr {{.*}}, i32 0, i32 0
// CHECK: %[[RM_HANDLE:.*]] = load target("dx.RawBuffer", [2 x <3 x float>], 1, 0), ptr %[[RM_HANDLE_PTR]], align 4
// CHECK: call ptr @llvm.dx.resource.getpointer{{.*}}(target("dx.RawBuffer", [2 x <3 x float>], 1, 0) %[[RM_HANDLE]], i32 {{.*}})

// CHECK-LABEL: define linkonce_odr hidden {{.*}} ptr @_ZNK4hlsl33RasterizerOrderedStructuredBuffer
// CHECK: %[[CM_DST_HANDLE_PTR:.*]] = getelementptr {{.*}}%"class.hlsl::RasterizerOrderedStructuredBuffer", ptr {{.*}}, i32 0, i32 0
// CHECK: %[[CM_DST_HANDLE:.*]] = load target("dx.RawBuffer", [3 x <2 x float>], 1, 1), ptr %[[CM_DST_HANDLE_PTR]], align 4
// CHECK: call ptr @llvm.dx.resource.getpointer{{.*}}(target("dx.RawBuffer", [3 x <2 x float>], 1, 1) %[[CM_DST_HANDLE]], i32 {{.*}})

// CHECK-LABEL: define linkonce_odr hidden {{.*}} ptr @_ZNK4hlsl16StructuredBufferIu11matrix_typeILm2ELm3ELm1EfEEixEj
// CHECK: %[[RM_SOURCE_HANDLE_PTR:.*]] = getelementptr {{.*}}%"class.hlsl::StructuredBuffer{{(\.0)?}}", ptr {{.*}}, i32 0, i32 0
// CHECK: %[[RM_SOURCE_HANDLE:.*]] = load target("dx.RawBuffer", [2 x <3 x float>], 0, 0), ptr %[[RM_SOURCE_HANDLE_PTR]], align 4
// CHECK: call ptr @llvm.dx.resource.getpointer{{.*}}(target("dx.RawBuffer", [2 x <3 x float>], 0, 0) %[[RM_SOURCE_HANDLE]], i32 {{.*}})
