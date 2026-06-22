// Test that declare target local variables are NOT affected by
// unified_shared_memory. Local variables always use direct access
// (no offload entry, no _decl_tgt_ref_ptr) regardless of USM. For
// comparison, enter variables with USM use pointer-reference indirection
// when normally they would also be direct access.
//
// CHECK lines not auto-generated because they are specifically verifying
// absence of ref ptr and offload entry for local variable and, by contrast,
// presence of ref ptr and offload entry for enter variable.

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEVICE

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp requires unified_shared_memory

int local_var;
#pragma omp declare target local(local_var)

int enter_var;
#pragma omp declare target enter(enter_var)

// local_var: direct access, no ref ptr, no offload entry
// HOST-DAG: @local_var = global i32 0
// HOST-NOT: @local_var_decl_tgt_ref_ptr

// enter_var with USM: pointer-reference indirection
// HOST-DAG: @enter_var_decl_tgt_ref_ptr = weak global ptr @enter_var
// HOST-DAG: @.offloading.entry.enter_var_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @enter_var_decl_tgt_ref_ptr, ptr @.offloading.entry_name{{.*}}, i64 8, i64 0, ptr null }, section "llvm_offload_entries"

// Device: local_var is a direct global, enter_var uses ref ptr
// DEVICE-DAG: @local_var = protected addrspace(1) global i32 0
// DEVICE-NOT: @local_var_decl_tgt_ref_ptr
// DEVICE-DAG: @enter_var_decl_tgt_ref_ptr = weak global ptr null

int use_vars() {
  int result = 0;
#pragma omp target map(from: result)
  {
    local_var = 42;
    enter_var = 10;
    result = local_var + enter_var;
  }
  return result;
}

#endif
