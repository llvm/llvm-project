// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck -check-prefixes=CUDA-DEVICE %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck -check-prefixes=HIP-DEVICE %s

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

// Confirm static global texture is externally visible and has a unique name. 
static texture<float, 2, ElementType> texRef;
//CUDA-DEVICE: @_ZL6texRef__static__{{.*}} = addrspace(1) externally_initialized global i64 undef, align 4
//HIP-DEVICE: @_ZL6texRef.static.{{.*}} = addrspace(1) externally_initialized global %struct.texture undef, align 4

struct v4f {
  float x, y, z, w;
};

__attribute__((device)) v4f tex2d_ld(texture<float, 2, ElementType>, float, float) asm("llvm.nvvm.texRef.unified.2d.v4f32.f32");

__attribute__((device)) float foo(float x, float y) {
  return tex2d_ld(texRef, x, y).x;
}
