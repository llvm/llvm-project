// Test that OpenCL kernel argument metadata preserves semantic address spaces
// even if the target has only one address space like x86_64 does.
// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple x86_64-unknown-linux-gnu -emit-cir -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

kernel void spir_addr_space_kernel_args(__global int *G, __constant int *C,
                                        __local int *L) {
  *G = *C + *L;
}

// CIR-LABEL: cir.func{{.*}} @spir_addr_space_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [#cir<lang_address_space(offload_global)>, #cir<lang_address_space(offload_constant)>, #cir<lang_address_space(offload_local)>]

kernel void global_device_host_kernel_args(
    __attribute__((opencl_global_device)) int *D,
    __attribute__((opencl_global_host)) int *H) {}

// CIR-LABEL: cir.func{{.*}} @global_device_host_kernel_args
// CIR-SAME: cir.cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<addr_space = [#cir<lang_address_space(offload_global_device)>, #cir<lang_address_space(offload_global_host)>]
