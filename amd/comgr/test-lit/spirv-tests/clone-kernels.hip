// REQUIRES: comgr-has-spirv-translator
// COM: Generate a SPIRV file from a HIP kernel
// RUN: %clang -x hip --offload-arch=amdgcnspirv -nogpulib -nogpuinc \
// RUN:    --no-gpu-bundle-output --offload-device-only -O3 %s -o %t.spv

// COM: Run Comgr Translator to covert SPIRV back to LLVM IR
// RUN: spirv-translator --block-sizes  "1024,512,256,64" %t.spv -o %t.translated.bc

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: %llvm-dis %t.translated.bc -o - | %FileCheck %s

// COM: Verify the expected kernels
// CHECK-DAG: define amdgpu_kernel void @kernel1
// CHECK-DAG: define amdgpu_kernel void @kernel1.bs512
// CHECK-DAG: define amdgpu_kernel void @kernel1.bs256
// CHECK-DAG: define amdgpu_kernel void @kernel1.bs64
// CHECK-DAG: define amdgpu_kernel void @kernel2
// CHECK-DAG: define amdgpu_kernel void @kernel2.bs64
// CHECK-DAG: define amdgpu_kernel void @kernel3
// CHECK-DAG: define amdgpu_kernel void @kernel3.bs512
// CHECK-DAG: define amdgpu_kernel void @kernel3.bs256
// CHECK-DAG: define amdgpu_kernel void @kernel4
// CHECK-DAG: define amdgpu_kernel void @kernel4.bs512
// CHECK-DAG: define amdgpu_kernel void @kernel4.bs256
// CHECK-DAG: define amdgpu_kernel void @kernel5

// COM: Verify that the unexpected variants are not there
// COM: Kernels should not have variants that match their maximum size (default
// COM: 1024)
// CHECK-NOT: define amdgpu_kernel void @kernel1.bs1024
// CHECK-NOT: define amdgpu_kernel void @kernel3.bs1024
// CHECK-NOT: define amdgpu_kernel void @kernel2.bs256
// CHECK-NOT: define amdgpu_kernel void @kernel4.bs728
// CHECK-NOT: define amdgpu_kernel void @kernel5.bs512
// COM: Kernels should not have variants larger than the maximum size
// CHECK-NOT: define amdgpu_kernel void @kernel2.bs1024
// CHECK-NOT: define amdgpu_kernel void @kernel2.bs512
// CHECK-NOT: define amdgpu_kernel void @kernel4.bs1024
// CHECK-NOT: define amdgpu_kernel void @kernel5.bs1024
// COM: Kernels should not have variants smaller than the minimum size
// FIXME: SPIR-V currently loses the minimum block size limit during
// FIXME: translation, so these variants are still generated:
// FIXME: kernel5.bs256, kernel3.bs64, kernel4.bs64 and kernel5.bs64

extern "C" {

__attribute__((global))
void kernel1(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(1, 256)))
void kernel2(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(256, 1024)))
void kernel3(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(128, 768)))
void kernel4(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(512, 512)))
void kernel5(float* a) {
    a[0] = 1.0f;
}

}
