// RUN: %clang_cc1 %s -finclude-default-header -fdeclare-opencl-builtins -cl-std=clc++ -fblocks -O0 -emit-llvm -o - -triple "spir-unknown-unknown" | FileCheck %s

void testBranchingOnEnqueueKernel(queue_t default_queue, unsigned flags, ndrange_t ndrange) {
    // Ensure `enqueue_kernel` can be branched upon.

    if (enqueue_kernel(default_queue, flags, ndrange, ^(void) {}))
        (void)0;
    // CHECK: [[P:%[0-9]+]] = call spir_func i32 @__enqueue_kernel
    // CHECK-NEXT: [[Q:%[a-z0-9]+]] = icmp ne i32 [[P]], 0
    // CHECK-NEXT: br i1 [[Q]]

    if (get_kernel_work_group_size(^(void) {}))
        (void)0;
    // CHECK: [[P:%[0-9]+]] = call spir_func i32 @__get_kernel_work_group_size
    // CHECK-NEXT: [[Q:%[a-z0-9]+]] = icmp ne i32 [[P]], 0
    // CHECK-NEXT: br i1 [[Q]]

    if (get_kernel_preferred_work_group_size_multiple(^(void) {}))
        (void)0;
    // CHECK: [[P:%[0-9]+]] = call spir_func i32 @__get_kernel_preferred_work_group_size_multiple_impl
    // CHECK-NEXT: [[Q:%[a-z0-9]+]] = icmp ne i32 [[P]], 0
    // CHECK-NEXT: br i1 [[Q]]
}

void testBranchinOnPipeOperations(read_only pipe int r, write_only pipe int w, global int* ptr) {
    // Verify that return type is correctly casted to i1 value.

    if (read_pipe(r, ptr))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__read_pipe_2
    // CHECK-NEXT: icmp ne i32 [[R]], 0

    if (write_pipe(w, ptr))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__write_pipe_2
    // CHECK-NEXT: icmp ne i32 [[R]], 0

    if (get_pipe_num_packets(r))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__get_pipe_num_packets_ro
    // CHECK-NEXT: icmp ne i32 [[R]], 0

    if (get_pipe_num_packets(w))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__get_pipe_num_packets_wo
    // CHECK-NEXT: icmp ne i32 [[R]], 0

    if (get_pipe_max_packets(r))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__get_pipe_max_packets_ro
    // CHECK-NEXT: icmp ne i32 [[R]], 0

    if (get_pipe_max_packets(w))
        (void)0;
    // CHECK: [[R:%[0-9]+]] = call spir_func i32 @__get_pipe_max_packets_wo
    // CHECK-NEXT: icmp ne i32 [[R]], 0
}

void testBranchingOnAddressSpaceCast(generic long* ptr) {
    // Verify that pointer types are properly casted, respecting address spaces.

    if (to_global(ptr))
        (void)0;
    // CHECK:       [[P:%[0-9]+]] = call spir_func [[GLOBAL_VOID:ptr addrspace\(1\)]] @__to_global([[GENERIC_VOID:ptr addrspace\(4\)]] {{%[0-9]+}})
    // CHECK-NEXT:  [[BOOL:%[a-z0-9]+]] = icmp ne ptr addrspace(1) [[P]], null
    // CHECK-NEXT:  br i1 [[BOOL]]

    if (to_local(ptr))
        (void)0;
    // CHECK:       [[P:%[0-9]+]] = call spir_func [[LOCAL_VOID:ptr addrspace\(3\)]] @__to_local([[GENERIC_VOID]] {{%[0-9]+}})
    // CHECK-NEXT:  [[BOOL:%[a-z0-9]+]] = icmp ne ptr addrspace(3) [[P]], null
    // CHECK-NEXT:  br i1 [[BOOL]]

    if (to_private(ptr))
        (void)0;
    // CHECK:       [[P:%[0-9]+]] = call spir_func [[PRIVATE_VOID:ptr]] @__to_private([[GENERIC_VOID]] {{%[0-9]+}})
    // CHECK-NEXT:  [[BOOL:%[a-z0-9]+]] = icmp ne ptr [[P]], null
    // CHECK-NEXT:  br i1 [[BOOL]]
}

