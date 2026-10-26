// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=CIR-DEVICE %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-DEVICE %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=OGCG-DEVICE %s

struct textureReference {
  int desc;
};

enum ReadMode {
  ElementType = 0,
  NormalizedFloat = 1
};

template <typename T, int dim = 1, ReadMode mode = ElementType>
struct __attribute__((device_builtin_texture_type)) texture
    : public textureReference {};

texture<float, 2, NormalizedFloat> tex;

// CIR-DEVICE: cir.global external target_address_space(1) @tex = #cir.undef : !cir.cuda_texture

// CIR now matches OG CodeGen and emits undef for CUDA shadow variables.
// LLVM-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
// OGCG-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
