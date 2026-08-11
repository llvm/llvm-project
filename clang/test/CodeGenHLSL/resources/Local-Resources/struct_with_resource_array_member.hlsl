// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -o - | FileCheck %s

// Mirror image of struct_array_with_resource_member.hlsl: there an array of
// structs each holding a resource, here a single struct holding an array of
// resources.

RWByteAddressBuffer GBuf : register(u0);

struct S { RWByteAddressBuffer Arr[2]; };

[numthreads(1,1,1)]
void main() {
    S Obj;
    Obj.Arr[0] = GBuf;
    Obj.Arr[0].Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// GBuf's handle (u0, space0) flows through the struct's array member into the
// Store of 42 at offset 0.
// CHECK: %[[H:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[P:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 0)
// CHECK: store i32 42, ptr %[[P]]
