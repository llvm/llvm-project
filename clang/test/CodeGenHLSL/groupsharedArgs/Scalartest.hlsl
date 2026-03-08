// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.2-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

groupshared uint16_t SharedData;

// CHECK-LABEL: define hidden void @_Z3fn1RU3AS3t(ptr addrspace(3) noundef align 2 dereferenceable(2) %Sh)
// CHECK: [[ShAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: store ptr addrspace(3) %Sh, ptr [[ShAddr]], align 4
// CHECK: [[A:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK: store i16 5, ptr addrspace(3) [[A]], align 2
// CHECK: ret void
void fn1(groupshared uint16_t Sh) {
  Sh = 5;
}

[numthreads(4, 1, 1)]
void main(uint3 TID : SV_GroupThreadID) {
  fn1(SharedData);
}
