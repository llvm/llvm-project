// RUN: %clang_cc1 -cl-std=CL2.0 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple spirv64-unknown-unknown -fexperimental-abi-lowering -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// OpenCL types are passed directly as their SPIR-V target extension types.
// CHECK: define spir_func void @f_img(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %i)
void f_img(read_only image2d_t i) {}
// CHECK: define spir_func void @f_smp(target("spirv.Sampler") %s)
void f_smp(sampler_t s) {}
// CHECK: define spir_func void @f_evt(target("spirv.Event") %e)
void f_evt(event_t e) {}
// CHECK: define spir_func void @f_q(target("spirv.Queue") %q)
void f_q(queue_t q) {}
// CHECK: define spir_func void @f_pipe(target("spirv.Pipe", 0) %p)
void f_pipe(read_only pipe int p) {}

// Kernels with OpenCL type arguments keep Clang's own ABI lowering, as the ABI
// library would coerce these like default address space pointers.
// CHECK: define spir_kernel void @k_img(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %i, ptr addrspace(1) {{.*}}%p)
kernel void k_img(read_only image2d_t i, global int *p) {}
// CHECK: define spir_kernel void @k_pipe(target("spirv.Pipe", 0) %p)
kernel void k_pipe(read_only pipe int p) {}
