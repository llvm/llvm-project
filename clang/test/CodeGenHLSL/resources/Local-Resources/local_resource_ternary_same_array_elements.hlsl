// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer In : register(u0);
RWByteAddressBuffer OutArr[] : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Out = true ? OutArr[0] : OutArr[1];
    Out.Store(0, In.Load(0));
}

// CHECK-LABEL: define {{.*}}@main(
// In (u0, space0) is loaded from; OutArr[0] (u1, space0) is stored to; the ternary always folds to OutArr[0].
// CHECK: %[[HI:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[HO:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// In.Load(0)
// CHECK: %[[PI:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HI]], i32 0)
// CHECK: %[[V:[^ ]+]] = load i32, ptr %[[PI]]
// OutArr[0].Store(0, In.Load(0))
// CHECK: %[[PO:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HO]], i32 0)
// CHECK: store i32 %[[V]], ptr %[[PO]]
