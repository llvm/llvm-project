// RUN: %clang_cc1 %s -fclangir -cl-std=CL2.0 -triple x86_64-unknown-linux-gnu -emit-cir -o - -verify

kernel void invalid_target_addr_space_kernel_arg(
    // expected-error@+1 {{ClangIR code gen Not Yet Implemented: OpenCL kernel argument metadata for target-specific address_space(N) kernel parameters; classic CodeGen currently accepts this case}}
    __attribute__((address_space(5))) int *T) {}
