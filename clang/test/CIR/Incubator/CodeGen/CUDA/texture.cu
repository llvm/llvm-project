// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE-LLVM %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=DEVICE-CIR %s
// RUN: echo "GPU binary would be here" > %t

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

texture<float, 2, NormalizedFloat> tex;

// DEVICE-LLVM: @tex = addrspace(1) externally_initialized global i64 undef, align 4
// DEVICE-CIR: cir.global external lang_address_space(offload_global) @tex = #cir.undef : !s64i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
