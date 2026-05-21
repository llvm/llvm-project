// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=DEVICE-CIR %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE-CIR-LLVM %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=DEVICE-OGCG %s

struct surfaceReference {
  int desc;
};

template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface
    : public surfaceReference {};

template <int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim>
    : public surfaceReference {};

surface<void, 2> surf;

// DEVICE-CIR: cir.global external target_address_space(1) @surf = #cir.poison : !s64i

// CIR lowers poison to LLVM poison, while OG CodeGen emits undef.
// DEVICE-CIR-LLVM: @surf ={{.*}} addrspace(1) externally_initialized global i64 poison
// DEVICE-OGCG: @surf ={{.*}} addrspace(1) externally_initialized global i64 undef