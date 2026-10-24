// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

template <typename T> static T Counter = T(1);
template <> int Counter<int> = 7;

template <typename T> static const T Scale[2] = {T(2), T(3)};

template <typename T> T Global;
template <typename T> uint Index;

template <typename T> groupshared T Shared[4];

struct S {
  template <typename T> static T Member;
};
template <typename T> T S::Member = T(5);

// DXIL-DAG: @_ZL7CounterIfE = internal global float 1.000000e+00, align 4
// DXIL-DAG: @_ZL7CounterIiE = internal global i32 7, align 4
// DXIL-DAG: @_ZL5ScaleIfE = internal constant [2 x float] [float 2.000000e+00, float 3.000000e+00], align 4
// DXIL-DAG: @_Z6GlobalIfE = external hidden addrspace(2) global float, align 4
// DXIL-DAG: @_Z5IndexIfE = external hidden addrspace(2) global i32, align 4
// DXIL-DAG: @_Z6SharedIfE = external hidden addrspace(3) global [4 x float], align 4
// DXIL-DAG: @_ZN1S6MemberIfEE = linkonce_odr hidden global float 5.000000e+00, align 4

// SPIRV-DAG: @_ZL7CounterIfE = internal addrspace(10) global float 1.000000e+00, align 4
// SPIRV-DAG: @_ZL7CounterIiE = internal addrspace(10) global i32 7, align 4
// SPIRV-DAG: @_ZL5ScaleIfE = internal addrspace(10) constant [2 x float] [float 2.000000e+00, float 3.000000e+00], align 4
// SPIRV-DAG: @_Z6GlobalIfE = external hidden addrspace(12) global float, align 4
// SPIRV-DAG: @_Z5IndexIfE = external hidden addrspace(12) global i32, align 4
// SPIRV-DAG: @_Z6SharedIfE = external hidden addrspace(3) global [4 x float], align 4
// SPIRV-DAG: @_ZN1S6MemberIfEE = linkonce_odr hidden addrspace(10) global float 5.000000e+00, comdat, align 4

// CHECK-LABEL: define {{.*}}float @_Z2fnj(
export float fn(uint I) {
// DXIL: store float {{.*}}, ptr @_ZL7CounterIfE, align 4
// SPIRV: store float {{.*}}, ptr addrspace(10) @_ZL7CounterIfE, align 4
  Counter<float> += 1.0;
// DXIL: load float, ptr addrspace(2) @_Z6GlobalIfE, align 4
// SPIRV: load float, ptr addrspace(12) @_Z6GlobalIfE, align 4
  Shared<float>[I] = Global<float>;
// DXIL: load i32, ptr @_ZL7CounterIiE, align 4
// SPIRV: load i32, ptr addrspace(10) @_ZL7CounterIiE, align 4
  return Counter<float> + Counter<int> + Scale<float>[I] + Shared<float>[I] +
         S::Member<float> + Index<float>;
}

// CHECK: !hlsl.cbs = !{![[CB:.*]]}
// DXIL: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @_Z6GlobalIfE, ptr addrspace(2) @_Z5IndexIfE}
// SPIRV: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(12) @_Z6GlobalIfE, ptr addrspace(12) @_Z5IndexIfE}
