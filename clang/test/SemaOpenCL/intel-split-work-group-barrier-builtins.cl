// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s
// expected-no-diagnostics

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int cl_mem_fence_flags;
typedef enum memory_scope {
  memory_scope_work_item = 0,
  memory_scope_work_group = 1,
  memory_scope_device = 2,
  memory_scope_all_svm_devices = 3,
  memory_scope_sub_group = 4
} memory_scope;

void test_split_work_group_barrier(cl_mem_fence_flags flags,
                                   memory_scope scope) {
  intel_work_group_barrier_arrive(flags);
  intel_work_group_barrier_wait(flags);
  intel_work_group_barrier_arrive(flags, scope);
  intel_work_group_barrier_wait(flags, scope);
}
