// RUN: %clang_dxc -T cs_6_3 -Xclang -emit-llvm -Xclang -disable-llvm-passes %s | FileCheck %s
// RUN: %clang_dxc -T cs_6_3 -Xclang -emit-llvm -Xclang -disable-llvm-passes -Xclang -load-bool-from-mem=truncate %s 2>&1 | FileCheck %s --check-prefix=CHECK,TRUNCATE

// TRUNCATE: -load-bool-from-mem=truncate ignored for HLSL (always nonzero)

StructuredBuffer<bool4> In : register(t0);
RWStructuredBuffer<bool4> Out : register(u1);

// CHECK-LABEL: define {{.*}} @_Z4mainv
// CHECK: [[Load:%.*]] = load <4 x i32>, ptr
// CHECK-NEXT: [[Elem:%.*]] = extractelement <4 x i32> [[Load]], i32 3
// CHECK-NEXT: [[Bool:%.*]] = icmp ne i32 [[Elem]], 0
// CHECK: [[Wide:%.*]] = zext i1 [[Bool]] to i32
// CHECK: [[Ptr:%.*]] = getelementptr <4 x i32>, ptr %{{.*}}, i32 0, i32 3
// CHECK-NEXT: store i32 [[Wide]], ptr [[Ptr]]
[numthreads(1,1,1)]
void main(void) {
    Out[1].w = In[0].w;
}

