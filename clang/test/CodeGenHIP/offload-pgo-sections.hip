// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// Verify the per-TU __llvm_profile_sections_<CUID> global for HIP+PGO.
// Device side: clang emits the names-postfix marker, and the InstrProfiling
// pass emits the populated 9-pointer struct in addrspace(1) -- but only when
// the TU actually has profile data records. Host compile: void* shadow
// registered with the HIP runtime and the profile runtime's drain list.

// The device struct is emitted by the InstrProfiling pass (not clang codegen),
// so run the pass to observe it.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | opt -passes=instrprof -S \
// RUN:   | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=HOST %s
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fgpu-rdc --offload-new-driver \
// RUN:   -cuid=abc -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=HOST-RDC %s

// Guard: no PGO -> no emission.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=NONE %s

// Guard: no CUID -> no emission.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefix=NONE %s

// Guard: PGO on but no instrumented device functions (all device code is
// constexpr/host-only) -> the pass must not emit the sections struct, so its
// section references don't dangle at the device link.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip \
// RUN:   -DEMPTY_DEVICE %s \
// RUN:   | opt -passes=instrprof -S \
// RUN:   | FileCheck -check-prefix=EMPTY %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

#ifdef EMPTY_DEVICE
// No __global__/instrumented device function: device code folds away.
__device__ constexpr int dead(int x) { return x + 1; }
#else
__device__ int helper(int x) { return x + 1; }
__global__ void kernel(int *p) { *p = helper(*p); }
#endif

// DEV-DAG: @__start___llvm_prf_names = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_names = external hidden addrspace(1) global i8
// DEV-DAG: @__start___llvm_prf_cnts = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_cnts = external hidden addrspace(1) global i8
// DEV-DAG: @__start___llvm_prf_data = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_data = external hidden addrspace(1) global i8
// DEV-DAG: @__start___llvm_prf_ucnts = external hidden addrspace(1) global i8
// DEV-DAG: @__stop___llvm_prf_ucnts = external hidden addrspace(1) global i8
// DEV-DAG: @__llvm_profile_raw_version = external addrspace(1) constant i64
// DEV-DAG: @__llvm_prf_nm_[[CUID:[0-9a-f]+]] = protected addrspace(1) constant {{.*}}section "__llvm_prf_names"
// DEV-DAG: @__llvm_profile_sections_[[CUID]] = protected addrspace(1) constant {{.*}}@__start___llvm_prf_names{{.*}}@__stop___llvm_prf_names{{.*}}@__start___llvm_prf_cnts{{.*}}@__stop___llvm_prf_cnts{{.*}}@__start___llvm_prf_data{{.*}}@__stop___llvm_prf_data{{.*}}@__start___llvm_prf_ucnts{{.*}}@__stop___llvm_prf_ucnts{{.*}}@__llvm_profile_raw_version
// DEV-DAG: @llvm.compiler.used = {{.*}}@__llvm_profile_sections_[[CUID]]

// HOST: @__llvm_profile_sections_[[CUID:[0-9a-f]+]] = global ptr null
// HOST-DAG: @__llvm_profile_shadow_data_[[CUID]]_{{[0-9]+}} = global ptr null
// HOST-DAG: @__llvm_profile_shadow_cnts_[[CUID]]_{{[0-9]+}} = global ptr null
// HOST-DAG: @__llvm_profile_shadow_ucnts_[[CUID]]_{{[0-9]+}} = global ptr null
// HOST-DAG: @__llvm_profile_shadow_names_[[CUID]]_{{[0-9]+}} = global ptr null
// HOST: @llvm.compiler.used = {{.*}}@__llvm_profile_sections_[[CUID]]
// HOST: define internal void @__hip_register_globals
// HOST: call void @__hipRegisterVar({{.*}}@__llvm_profile_sections_[[CUID]],
// HOST: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_[[CUID]])
// HOST-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_data_[[CUID]]_{{[0-9]+}})
// HOST-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_cnts_[[CUID]]_{{[0-9]+}})
// HOST-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_ucnts_[[CUID]]_{{[0-9]+}})
// HOST: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_names_[[CUID]]_{{[0-9]+}})

// HOST-RDC: @__llvm_profile_sections_[[CUID:[0-9a-f]+]] = global ptr null
// HOST-RDC-DAG: @__llvm_profile_shadow_data_[[CUID]]_0 = global ptr null
// HOST-RDC-DAG: @__llvm_profile_shadow_cnts_[[CUID]]_1 = global ptr null
// HOST-RDC-DAG: @__llvm_profile_shadow_ucnts_[[CUID]]_2 = global ptr null
// HOST-RDC-DAG: @__llvm_profile_shadow_names_[[CUID]]_3 = global ptr null
// HOST-RDC-DAG: @.offloading.entry.{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_sections_[[CUID]]
// HOST-RDC-DAG: @.offloading.entry.{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_data_[[CUID]]_0
// HOST-RDC-DAG: @.offloading.entry.{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_cnts_[[CUID]]_1
// HOST-RDC-DAG: @.offloading.entry.{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_ucnts_[[CUID]]_2
// HOST-RDC-DAG: @.offloading.entry.{{.*}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_names_[[CUID]]_3
// HOST-RDC: define internal void @__llvm_profile_register_shadow.[[CUID]]()
// HOST-RDC: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_[[CUID]])
// HOST-RDC-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_data_[[CUID]]_0)
// HOST-RDC-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_cnts_[[CUID]]_1)
// HOST-RDC-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_ucnts_[[CUID]]_2)
// HOST-RDC-DAG: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_names_[[CUID]]_3)

// NONE-NOT: __llvm_profile_sections_
// NONE-NOT: __llvm_profile_offload_register_shadow_variable

// EMPTY-NOT: @__llvm_profile_sections_
// EMPTY-NOT: @__start___llvm_prf_data
