#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-LABEL: @_Z19test_atom_add_gen_iPii
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_add_gen_iPii
// LLVM: atomicrmw add ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_add_gen_i(int *p, int val) {
  __nvvm_atom_add_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_add_gen_lPll
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_add_gen_lPll
// LLVM: atomicrmw add ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_add_gen_l(long *p, long val) {
  __nvvm_atom_add_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_add_gen_llPxx
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_add_gen_llPxx
// LLVM: atomicrmw add ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_add_gen_ll(long long *p, long long val) {
  __nvvm_atom_add_gen_ll(p, val);
}
