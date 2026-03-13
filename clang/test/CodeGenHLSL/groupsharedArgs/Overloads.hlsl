// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

// Verify we are calling the correct overloads
void fn(groupshared float4 Arr[2]);
void fn(inout float4 Arr[2]);

void fn2(groupshared int4 Shared);
void fn2(int4 Local);

[numthreads(4,1,1)]
void main() {
  float4 Local[2] = {1.0.xxxx, 2.0.xxxx};
// CHECK: call void @_Z2fnA2_Dv4_f
  fn(Local);

// CHECK: call void @_Z3fn2Dv4_i
  fn2(11.xxxx);
}

// CHECK-LABEL: define hidden void @_Z2fnA2_Dv4_f(ptr noalias noundef align 16 %Arr)
void fn(inout float4 Arr[2]) {
  Arr[1] = 5.0.xxxx;
}

// CHECK-LABEL: define hidden void @_Z3fn2Dv4_i(<4 x i32> noundef %Local) #0 {
void fn2(int4 Local) {
  int X = Local.y;
}

// CHECK-LABEL: define hidden void @_Z2fnRA2_U3AS3Dv4_f(ptr addrspace(3) noundef align 16 dereferenceable(32) %Arr)
void fn(groupshared float4 Arr[2]) {
  Arr[1] = 7.0.xxxx;
}

// CHECK-LABEL: define hidden void @_Z3fn2RU3AS3Dv4_i(ptr addrspace(3) noundef align 16 dereferenceable(16) %Shared)
void fn2(groupshared int4 Shared) {
  Shared.x = 10;
}
