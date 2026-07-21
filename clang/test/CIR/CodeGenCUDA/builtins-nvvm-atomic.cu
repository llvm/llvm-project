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

// CIR-LABEL: @_Z19test_atom_sub_gen_iPii
// CIR: cir.atomic.fetch sub relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_sub_gen_iPii
// LLVM: atomicrmw sub ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_sub_gen_i(int *p, int val) {
  __nvvm_atom_sub_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_sub_gen_lPll
// CIR: cir.atomic.fetch sub relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_sub_gen_lPll
// LLVM: atomicrmw sub ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_sub_gen_l(long *p, long val) {
  __nvvm_atom_sub_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_sub_gen_llPxx
// CIR: cir.atomic.fetch sub relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_sub_gen_llPxx
// LLVM: atomicrmw sub ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_sub_gen_ll(long long *p, long long val) {
  __nvvm_atom_sub_gen_ll(p, val);
}

// CIR-LABEL: @_Z19test_atom_and_gen_iPii
// CIR: cir.atomic.fetch and relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_and_gen_iPii
// LLVM: atomicrmw and ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_and_gen_i(int *p, int val) {
  __nvvm_atom_and_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_and_gen_lPll
// CIR: cir.atomic.fetch and relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_and_gen_lPll
// LLVM: atomicrmw and ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_and_gen_l(long *p, long val) {
  __nvvm_atom_and_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_and_gen_llPxx
// CIR: cir.atomic.fetch and relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_and_gen_llPxx
// LLVM: atomicrmw and ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_and_gen_ll(long long *p, long long val) {
  __nvvm_atom_and_gen_ll(p, val);
}

// CIR-LABEL: @_Z18test_atom_or_gen_iPii
// CIR: cir.atomic.fetch or relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z18test_atom_or_gen_iPii
// LLVM: atomicrmw or ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_or_gen_i(int *p, int val) {
  __nvvm_atom_or_gen_i(p, val);
}

// CIR-LABEL: @_Z18test_atom_or_gen_lPll
// CIR: cir.atomic.fetch or relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z18test_atom_or_gen_lPll
// LLVM: atomicrmw or ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_or_gen_l(long *p, long val) {
  __nvvm_atom_or_gen_l(p, val);
}

// CIR-LABEL: @_Z19test_atom_or_gen_llPxx
// CIR: cir.atomic.fetch or relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_or_gen_llPxx
// LLVM: atomicrmw or ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_or_gen_ll(long long *p, long long val) {
  __nvvm_atom_or_gen_ll(p, val);
}

// CIR-LABEL: @_Z19test_atom_xor_gen_iPii
// CIR: cir.atomic.fetch xor relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_xor_gen_iPii
// LLVM: atomicrmw xor ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_xor_gen_i(int *p, int val) {
  __nvvm_atom_xor_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_xor_gen_lPll
// CIR: cir.atomic.fetch xor relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_xor_gen_lPll
// LLVM: atomicrmw xor ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_xor_gen_l(long *p, long val) {
  __nvvm_atom_xor_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_xor_gen_llPxx
// CIR: cir.atomic.fetch xor relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_xor_gen_llPxx
// LLVM: atomicrmw xor ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_xor_gen_ll(long long *p, long long val) {
  __nvvm_atom_xor_gen_ll(p, val);
}

// CIR-LABEL: @_Z19test_atom_max_gen_iPii
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_max_gen_iPii
// LLVM: atomicrmw max ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_max_gen_i(int *p, int val) {
  __nvvm_atom_max_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_max_gen_lPll
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_max_gen_lPll
// LLVM: atomicrmw max ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_max_gen_l(long *p, long val) {
  __nvvm_atom_max_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_max_gen_llPxx
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_max_gen_llPxx
// LLVM: atomicrmw max ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_max_gen_ll(long long *p, long long val) {
  __nvvm_atom_max_gen_ll(p, val);
}

// CIR-LABEL: @_Z20test_atom_max_gen_uiPjj
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u32i>, !u32i) -> !u32i
// LLVM-LABEL: @_Z20test_atom_max_gen_uiPjj
// LLVM: atomicrmw umax ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_max_gen_ui(unsigned *p, unsigned val) {
  __nvvm_atom_max_gen_ui(p, val);
}

// CIR-LABEL: @_Z20test_atom_max_gen_ulPmm
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u64i>, !u64i) -> !u64i
// LLVM-LABEL: @_Z20test_atom_max_gen_ulPmm
// LLVM: atomicrmw umax ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_max_gen_ul(unsigned long *p, unsigned long val) {
  __nvvm_atom_max_gen_ul(p, val);
}

// CIR-LABEL: @_Z21test_atom_max_gen_ullPyy
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u64i>, !u64i) -> !u64i
// LLVM-LABEL: @_Z21test_atom_max_gen_ullPyy
// LLVM: atomicrmw umax ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_max_gen_ull(unsigned long long *p, unsigned long long val) {
  __nvvm_atom_max_gen_ull(p, val);
}

// CIR-LABEL: @_Z19test_atom_min_gen_iPii
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
// LLVM-LABEL: @_Z19test_atom_min_gen_iPii
// LLVM: atomicrmw min ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_min_gen_i(int *p, int val) {
  __nvvm_atom_min_gen_i(p, val);
}

// CIR-LABEL: @_Z19test_atom_min_gen_lPll
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z19test_atom_min_gen_lPll
// LLVM: atomicrmw min ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_min_gen_l(long *p, long val) {
  __nvvm_atom_min_gen_l(p, val);
}

// CIR-LABEL: @_Z20test_atom_min_gen_llPxx
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !s64i) -> !s64i
// LLVM-LABEL: @_Z20test_atom_min_gen_llPxx
// LLVM: atomicrmw min ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_min_gen_ll(long long *p, long long val) {
  __nvvm_atom_min_gen_ll(p, val);
}

// CIR-LABEL: @_Z20test_atom_min_gen_uiPjj
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u32i>, !u32i) -> !u32i
// LLVM-LABEL: @_Z20test_atom_min_gen_uiPjj
// LLVM: atomicrmw umin ptr %{{.*}}, i32 %{{.*}} monotonic, align 4
__device__ void test_atom_min_gen_ui(unsigned *p, unsigned val) {
  __nvvm_atom_min_gen_ui(p, val);
}

// CIR-LABEL: @_Z20test_atom_min_gen_ulPmm
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u64i>, !u64i) -> !u64i
// LLVM-LABEL: @_Z20test_atom_min_gen_ulPmm
// LLVM: atomicrmw umin ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_min_gen_ul(unsigned long *p, unsigned long val) {
  __nvvm_atom_min_gen_ul(p, val);
}

// CIR-LABEL: @_Z21test_atom_min_gen_ullPyy
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.*}}, %{{.*}} : (!cir.ptr<!u64i>, !u64i) -> !u64i
// LLVM-LABEL: @_Z21test_atom_min_gen_ullPyy
// LLVM: atomicrmw umin ptr %{{.*}}, i64 %{{.*}} monotonic, align 8
__device__ void test_atom_min_gen_ull(unsigned long long *p, unsigned long long val) {
  __nvvm_atom_min_gen_ull(p, val);
}
