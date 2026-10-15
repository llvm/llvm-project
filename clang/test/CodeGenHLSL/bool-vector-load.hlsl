// RUN: %clang_dxc -T cs_6_6 %s | FileCheck %s
// RUN: %clang_dxc -T cs_6_6 -Xclang -load-bool-from-mem=truncate %s 2>&1 | FileCheck %s --check-prefix=CHECK,TRUNCATE

// TRUNCATE: -load-bool-from-mem=truncate ignored for HLSL (always nonzero)

StructuredBuffer<bool4> In : register(t0);
RWStructuredBuffer<bool4> Out : register(u1);

// CHECK-LABEL: define {{.*}}main
// CHECK: [[Load:%.*]] = call {{[^ ]*}} @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %{{[^,]*}}, i32 0, i32 12, i8 1, i32 4)
// CHECK: [[Elem:%.*]] = extractvalue {{.*}} [[Load]], 0
// CHECK-NEXT: [[Bool:%.*]] = icmp ne i32 [[Elem]], 0
// CHECK: [[Wide:%.*]] = zext i1 [[Bool]] to i32
// CHECK: call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %{{[^,]*}},{{[^,]*}},{{[^,]*}}, i32 [[Wide]]
[numthreads(1,1,1)]
void main(void) {
    Out[1].w = In[0].w;
}

