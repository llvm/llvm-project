// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple spir64 -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=SPIR %s
// RUN: %clang_cc1 -triple spir64 -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=SPIR %s

// SPIR-DAG: @tab = externally_initialized constant [2 x i32]
// SPIR-DAG: @dev = externally_initialized global i32 3
// SPIR-DAG: @_ZZ10static_fmtiE3fmt = internal constant [3 x i8]
// SPIR-DAG: @_ZZ10get_sharedvE2sh = internal global [4 x i32]

#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))
#define __shared__ __attribute__((shared))

// CIR-DAG: cir.global {{.*}} target_address_space(1) @tab
// CIR-DAG: cir.global {{.*}} target_address_space(1) @_ZZ10static_fmtiE3fmt
// LLVM-DAG: @tab = addrspace(1) externally_initialized constant [2 x i32] [i32 1, i32 2]
// CIR-DAG: cir.global {{.*}} target_address_space(1) @dev
// CIR-DAG: cir.global {{.*}} target_address_space(3) @_ZZ10get_sharedvE2sh
// LLVM-DAG: @_ZZ10static_fmtiE3fmt = internal addrspace(1) constant [3 x i8] c"%d\00"
// LLVM-DAG: @dev = addrspace(1) externally_initialized global i32 3
// LLVM-DAG: @_ZZ10get_sharedvE2sh = internal addrspace(3) global [4 x i32]
// LLVM-DAG: @_Z5aliasv = alias i32 (), ptr addrspace(4) @alias_target
__constant__ int tab[2] = {1, 2};
__device__ int dev = 3;

// Functions live in the program address space 4.
// LLVM-DAG: @llvm.used = {{.*}}ptr addrspace(4) @_Z7used_fnv
__attribute__((used)) __device__ void used_fn() {}

// CIR-LABEL: cir.func{{.*}} @_Z10static_fmti
// CIR: %[[G:.*]] = cir.get_global @_ZZ10static_fmtiE3fmt : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)> -> !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>

// LLVM-LABEL: define{{.*}} ptr addrspace(4) @_Z10static_fmti
// LLVM: addrspacecast (ptr addrspace(1) @_ZZ10static_fmtiE3fmt to ptr addrspace(4))
__device__ const char *static_fmt(int i) {
  static const char fmt[] = "%d";
  return fmt + i;
}

// CIR-LABEL: cir.func{{.*}} @_Z7get_tabv
// CIR: %[[G:.*]] = cir.get_global @tab : !cir.ptr<!cir.array<!s32i x 2>, target_address_space(1)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s32i x 2>, target_address_space(1)> -> !cir.ptr<!cir.array<!s32i x 2>, target_address_space(4)>

// LLVM-LABEL: define{{.*}} ptr addrspace(4) @_Z7get_tabv
// LLVM: addrspacecast (ptr addrspace(1) @tab to ptr addrspace(4))
__device__ const int *get_tab() { return tab; }

__device__ int get_shared() {
  __shared__ int sh[4];
  return dev + sh[0];
}

// An explicit opencl_constant qualifier keeps UniformConstant.
// LLVM-LABEL: define{{.*}} i32 @_Z6get_ocPU3AS2i(ptr addrspace(2)
__device__ int get_oc(__attribute__((opencl_constant)) int *p) { return *p; }

// Function aliases point into the program address space 4.
extern "C" __device__ int alias_target() { return 1; }
[[gnu::alias("alias_target")]] __device__ int alias();

// Host-only aliases are skipped on the device.
extern "C" int host_target() { return 2; }
[[gnu::alias("host_target")]] int host_alias();
