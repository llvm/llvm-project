// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE-LLVM %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=DEVICE-CIR %s
// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

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

surface<void, 2> surf;

// DEVICE-LLVM: @surf = addrspace(1) externally_initialized global i64 undef, align 4
// DEVICE-CIR: cir.global external lang_address_space(offload_global) @surf = #cir.undef : !s64i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// HOST: @surf = global %"struct.surface<void, 2>" zeroinitializer, align 4