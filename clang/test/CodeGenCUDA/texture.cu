// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE %s
// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

struct textureReference {
  int desc;
};

enum ReadMode {
  ElementType = 0,
  NormalizedFloat = 1
};

template <typename T, int dim = 1, enum ReadMode mode = ElementType>
struct __attribute__((device_builtin_texture_type)) texture : public textureReference {
};

// On the device side, texture references are represented as `i64` handles.
// DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef, align 4
// DEVICE: @norm ={{.*}} addrspace(1) externally_initialized global i64 undef, align 4
// On the host side, they remain in the original type.
// HOST: @tex = internal global %struct.texture
// HOST: @norm = internal global %struct.texture
// HOST: @0 = private unnamed_addr constant [4 x i8] c"tex\00"
// HOST: @1 = private unnamed_addr constant [5 x i8] c"norm\00"
texture<float, 2, ElementType> tex;
texture<float, 2, NormalizedFloat> norm;

struct v4f {
  float x, y, z, w;
};

__attribute__((device)) v4f tex2d_ld(texture<float, 2, ElementType>, float, float) asm("llvm.nvvm.tex.unified.2d.v4f32.f32");
__attribute__((device)) v4f tex2d_ld(texture<float, 2, NormalizedFloat>, int, int) asm("llvm.nvvm.tex.unified.2d.v4f32.s32");

// DEVICE-LABEL: float @_Z3fooff(float %x, float %y)
// DEVICE: call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @tex)
// DEVICE: call %struct.v4f @llvm.nvvm.tex.unified.2d.v4f32.f32(i64 %{{.*}}, float %{{.*}}, float %{{.*}})
// DEVICE: call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @norm)
// DEVICE: call %struct.v4f @llvm.nvvm.tex.unified.2d.v4f32.s32(i64 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
__attribute__((device)) float foo(float x, float y) {
  return tex2d_ld(tex, x, y).x + tex2d_ld(norm, int(x), int(y)).x;
}

// HOST: define internal void @[[PREFIX:__cuda]]_register_globals
// Texture references need registering with correct arguments.
// HOST: call void @[[PREFIX]]RegisterTexture(i8** %0, i8*{{.*}}({{.*}}@tex{{.*}}), i8*{{.*}}({{.*}}@0{{.*}}), i8*{{.*}}({{.*}}@0{{.*}}), i32 2, i32 0, i32 0)
// HOST: call void @[[PREFIX]]RegisterTexture(i8** %0, i8*{{.*}}({{.*}}@norm{{.*}}), i8*{{.*}}({{.*}}@1{{.*}}), i8*{{.*}}({{.*}}@1{{.*}}), i32 2, i32 1, i32 0)

// They also need annotating in metadata.
// DEVICE: !0 = !{i64 addrspace(1)* @tex, !"texture", i32 1}
// DEVICE: !1 = !{i64 addrspace(1)* @norm, !"texture", i32 1}
