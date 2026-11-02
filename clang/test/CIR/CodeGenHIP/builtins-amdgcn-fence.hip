// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o %t.amd.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.amd.cir
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o %t.amd-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,AMD %s --input-file=%t.amd-cir.ll
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.amd.ll
// RUN: FileCheck --check-prefixes=LLVM,AMD %s --input-file=%t.amd.ll
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o %t.spv.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.spv.cir
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o %t.spv-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,SPV %s --input-file=%t.spv-cir.ll
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.spv.ll
// RUN: FileCheck --check-prefixes=LLVM,SPV %s --input-file=%t.spv.ll

#define __device__ __attribute__((device))

// CIR-DAG: #[[MMRA_G:.*]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">
// CIR-DAG: #[[MMRA_L:.*]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

// CIR-LABEL: @_Z6scopesv
// LLVM-LABEL: @_Z6scopesv
__device__ void scopes() {
  // CIR: cir.atomic.fence syncscope(device) seq_cst
  // AMD: fence syncscope("agent") seq_cst
  // SPV: fence syncscope("device") seq_cst
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
  // CIR: cir.atomic.fence syncscope(workgroup) seq_cst
  // LLVM: fence syncscope("workgroup") seq_cst
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
  // CIR: cir.atomic.fence syncscope(wavefront) acq_rel
  // AMD: fence syncscope("wavefront") acq_rel
  // SPV: fence syncscope("subgroup") acq_rel
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
  // CIR: cir.atomic.fence syncscope(single_thread) acquire
  // LLVM: fence syncscope("singlethread") acquire
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "singlethread");
  // CIR: cir.atomic.fence syncscope(cluster) release
  // AMD: fence syncscope("cluster") release
  // SPV: fence syncscope("workgroup") release
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
  // CIR: cir.atomic.fence syncscope(system) seq_cst
  // LLVM: fence seq_cst
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

// CIR-LABEL: @_Z9addrspacev
// LLVM-LABEL: @_Z9addrspacev
__device__ void addrspace() {
  // CIR: cir.atomic.fence syncscope(workgroup) release {llvm.mmra = #[[MMRA_G]]}
  // LLVM: fence syncscope("workgroup") release, !mmra ![[GLOBAL:[0-9]+]]
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
  // CIR: cir.atomic.fence syncscope(workgroup) acquire {llvm.mmra = #[[MMRA_L]]}
  // LLVM: fence syncscope("workgroup") acquire, !mmra ![[LOCAL:[0-9]+]]
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
  // CIR: cir.atomic.fence syncscope(device) seq_cst {llvm.mmra = [#[[MMRA_G]], #[[MMRA_L]]]}
  // AMD: fence syncscope("agent") seq_cst, !mmra ![[BOTH:[0-9]+]]
  // SPV: fence syncscope("device") seq_cst, !mmra ![[BOTH:[0-9]+]]
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent", "local", "global");
}

// LLVM-DAG: ![[GLOBAL]] = !{!"amdgpu-synchronize-as", !"global"}
// LLVM-DAG: ![[LOCAL]] = !{!"amdgpu-synchronize-as", !"local"}
// LLVM-DAG: ![[BOTH]] = !{![[GLOBAL]], ![[LOCAL]]}
