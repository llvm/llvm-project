// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=concurrency -nogpuinc \
// RUN:     -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "-triple" "amdgpu9.08-amd-amdhsa"{{.*}}"-fsanitize=concurrency"
// CHECK-OPENMP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=concurrency"
// CHECK-OPENMP-DAG: "-u" "__csan_offload_init"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=concurrency -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP
// CHECK-HIP-DAG: "-triple" "amdgpu9.08-amd-amdhsa"{{.*}}"-fsanitize=concurrency"
// CHECK-HIP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=concurrency"
// CHECK-HIP-DAG: "-u" "__csan_offload_init"
// CHECK-HIP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"
// CHECK-HIP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "-u" "__csan_offload_init"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// The device toolchain provides UBSan but not CSan.
// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=concurrency -nogpuinc \
// RUN:     -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-DEVICE-RT
// CHECK-NO-DEVICE-RT-NOT: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=concurrency"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp -fopenmp-targets=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=concurrency \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OMP-CPU
// CHECK-OMP-CPU-NOT: csan_offload
// CHECK-OMP-CPU-NOT: __csan_offload_init

int main(void) { return 0; }
