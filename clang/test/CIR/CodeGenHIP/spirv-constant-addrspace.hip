// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))

// CIR-DAG: cir.global {{.*}} target_address_space(1) @tab
// CIR-DAG: cir.global {{.*}} target_address_space(1) @_ZZ10static_fmtiE3fmt
// LLVM-DAG: @tab = addrspace(1) externally_initialized constant [2 x i32] [i32 1, i32 2]
// LLVM-DAG: @_ZZ10static_fmtiE3fmt = internal addrspace(1) constant [3 x i8] c"%d\00"
__constant__ int tab[2] = {1, 2};

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

// An explicit opencl_constant qualifier keeps UniformConstant.
// LLVM-LABEL: define{{.*}} i32 @_Z6get_ocPU3AS2i(ptr addrspace(2)
__device__ int get_oc(__attribute__((opencl_constant)) int *p) { return *p; }
