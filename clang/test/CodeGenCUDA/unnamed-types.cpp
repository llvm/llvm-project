// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -aux-triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN: -x cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -aux-triple nvptx64-nvidia-cuda  \
// RUN: -x cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST

int cudaConfigureCall(int, int, decltype(sizeof(int)) = 0, void* = nullptr);
namespace QL {
auto dg1 = [] { return 1; };
}
namespace QL {
auto dg2 = [] { return 2; };
}
using namespace QL;
template<typename T>
__attribute__((global)) void f(T t) {
  t();
}
void g() {
  f<<<1,1>>>(dg1);
  f<<<1,1>>>(dg2);
}

// HOST: @_ZN2QL3dg1E = internal global %class.anon undef, align 1
// HOST: @_ZN2QL3dg2E = internal global %class.anon.0 undef, align 1

// DEVICE: define void @_Z1fIN2QL3dg1MUlvE_EEvT_
// DEVICE: call noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// DEVICE: define internal noundef i32 @_ZNK2QL3dg1MUlvE_clEv
// DEVICE: define void @_Z1fIN2QL3dg2MUlvE_EEvT_
// DEVICE: call noundef i32 @_ZNK2QL3dg2MUlvE_clEv
// DEVICE: define internal noundef i32 @_ZNK2QL3dg2MUlvE_clEv

// HOST: define dso_local void @_Z1gv
// HOST: call void @_Z16__device_stub__fIN2QL3dg1MUlvE_EEvT_
// HOST: call void @_Z16__device_stub__fIN2QL3dg2MUlvE_EEvT_
// HOST: define internal void @_Z16__device_stub__fIN2QL3dg1MUlvE_EEvT_
// HOST: define internal void @_Z16__device_stub__fIN2QL3dg2MUlvE_EEvT_
