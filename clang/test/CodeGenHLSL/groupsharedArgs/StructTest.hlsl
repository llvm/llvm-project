// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

struct Shared {
  int A;
  float F;
  double Arr[1];
};

// CHECK: [[SharedData:@.*]] = external hidden addrspace(3) global %struct.Shared, align 1
groupshared Shared SharedData;
// CHECK: [[SharedData2:@.*]] = external hidden addrspace(3) global %struct.Shared, align 1
groupshared Shared SharedData2;

// CHECK-LABEL: define hidden void @_Z3fn1RU3AS36Shared(ptr addrspace(3) noundef align 1 dereferenceable(16) %Sh)
// CHECK: [[ShAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK-NEXT: [[DAddr:%.*]] = alloca double, align 8
// CHECK-NEXT: store ptr addrspace(3) %Sh, ptr [[ShAddr]], align 4
// CHECK-NEXT: [[Sh:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK-NEXT: [[A:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr addrspace(3) [[Sh]], i32 0, i32 0
// CHECK-NEXT: store i32 10, ptr addrspace(3) [[A]], align 1
// CHECK-NEXT: [[Sh2:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK-NEXT: [[F:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr addrspace(3) [[Sh2]], i32 0, i32 1
// CHECK-NEXT: store float 0x40263851E0000000, ptr addrspace(3) [[F]], align 1
// CHECK-NEXT: store double 1.000000e+01, ptr [[DAddr]], align 8
// CHECK-NEXT: [[D:%.*]] = load double, ptr [[DAddr]], align 8
// CHECK-NEXT: [[Sh3:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK-NEXT: [[Arr:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr addrspace(3) [[Sh3]], i32 0, i32 2
// CHECK-NEXT: [[ArrIdx:%.*]] = getelementptr inbounds [1 x double], ptr addrspace(3) [[Arr]], i32 0, i32 1
// CHECK-NEXT: store double [[D]], ptr addrspace(3) [[ArrIdx]], align 1
// CHECK-NEXT: ret void
void fn1(groupshared Shared Sh) {
  Sh.A = 10;
  Sh.F = 11.11;
  double D = 10.0;
  Sh.Arr[1] = D;
}

// CHECK-LABEL: define internal void @_Z4mainDv3_j(<3 x i32> noundef %TID)
[numthreads(4, 1, 1)]
void main(uint3 TID : SV_GroupThreadID) {
// CHECK: [[SAddr:%.*]] = alloca %struct.Shared, align 1
// CHECK: call void @_Z3fn1RU3AS36Shared(ptr addrspace(3) noundef align 1 dereferenceable(16) [[SharedData]])
  fn1(SharedData);

// CHECK-NEXT: [[A:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr [[SAddr]], i32 0, i32 0
// CHECK-NEXT: [[SD:%.*]] = load i32, ptr addrspace(3) [[SharedData]], align 1
// CHECK-NEXT: store i32 [[SD]], ptr [[A]], align 1
// CHECK-NEXT: [[F:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr [[SAddr]], i32 0, i32 1
// CHECK-NEXT: [[F2:%.*]] = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) [[SharedData]], i32 4), align 1
// CHECK-NEXT: store float [[F2]], ptr [[F]], align 1
// CHECK-NEXT: [[Arr:%.*]] = getelementptr inbounds nuw %struct.Shared, ptr [[SAddr]], i32 0, i32 2
// CHECK-NEXT: [[Arr2:%.*]] = load double, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) [[SharedData]], i32 8), align 1
// CHECK-NEXT: store double [[Arr2]], ptr [[Arr]], align 1
  Shared S = SharedData;

// CHECK-NEXT: call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 1 @SharedData2, ptr addrspace(3) align 1 @SharedData, i32 16, i1 false)
  SharedData2 = SharedData;
}
