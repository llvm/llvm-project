// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE %s
// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

struct surfaceReference {
  int desc;
};

template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface : public surfaceReference {
};

// Partial specialization over `void`.
template<int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim> : public surfaceReference {
};

// On the device side, surface references are represented as `i64` handles.
// DEVICE: @surf ={{.*}} addrspace(1) externally_initialized global i64 undef, align 4
// On the host side, they remain in the original type.
// HOST: @surf = internal global %struct.surface
// HOST: @0 = private unnamed_addr constant [5 x i8] c"surf\00"
surface<void, 2> surf;

__attribute__((device)) int suld_2d_zero(surface<void, 2>, int, int) asm("llvm.nvvm.suld.2d.i32.zero");

// DEVICE-LABEL: i32 @_Z3fooii(i32 noundef %x, i32 noundef %y)
// DEVICE: call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @surf)
// DEVICE: call noundef i32 @llvm.nvvm.suld.2d.i32.zero(i64 %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
__attribute__((device)) int foo(int x, int y) {
  return suld_2d_zero(surf, x, y);
}

__attribute__((device)) int musttail_callee(surface<void, 2> s, int x);

// A surface passed to a musttail call must keep the handle materialization;
// the argument is not forwarded as a raw lvalue.
// DEVICE-LABEL: @_Z15musttail_caller7surfaceIvLi2EEi(
// DEVICE: call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @surf)
// DEVICE: musttail call noundef i32 @_Z15musttail_callee7surfaceIvLi2EEi(i64
__attribute__((device)) int musttail_caller(surface<void, 2> s, int x) {
  [[clang::musttail]] return musttail_callee(surf, x);
}

// HOST: define internal void @[[PREFIX:__cuda]]_register_globals
// Texture references need registering with correct arguments.
// HOST: call void @[[PREFIX]]RegisterSurface(ptr %0, ptr @surf, ptr @0, ptr @0, i32 2, i32 0)

// They also need annotating in metadata.
// DEVICE: !0 = !{ptr addrspace(1) @surf, !"surface", i32 1}
