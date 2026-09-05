// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=undefined -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=undefined"
// CHECK-OPENMP-DAG: "-u" "__ubsan_offload_init"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=undefined -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP-HOST
// CHECK-HIP-HOST-DAG: "-u" "__ubsan_offload_init"
// CHECK-HIP-HOST-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=undefined \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "-u" "__ubsan_offload_init"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_standalone.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_host -fsanitize=undefined \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-HOST
// CHECK-XARCH-HOST-NOT: ubsan_offload
// CHECK-XARCH-HOST-NOT: __ubsan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HOST
// CHECK-HOST-NOT: ubsan_offload
// CHECK-HOST-NOT: __ubsan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=undefined \
// RUN:     -fsanitize-minimal-runtime -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MINIMAL
// CHECK-MINIMAL-NOT: ubsan_offload
// CHECK-MINIMAL-NOT: __ubsan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=undefined \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-DEV
// CHECK-SHARED-DEV-DAG: "-u" "__ubsan_offload_init"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_standalone.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=undefined \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED
// CHECK-SHARED-DAG: "-u" "__ubsan_offload_init"
// CHECK-SHARED-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_gfx908 -fsanitize=undefined \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-GPU
// CHECK-XARCH-GPU-DAG: "-u" "__ubsan_offload_init"
// CHECK-XARCH-GPU-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.ubsan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_gfx90a -fsanitize=undefined \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-OTHER
// CHECK-XARCH-OTHER-NOT: ubsan_offload
// CHECK-XARCH-OTHER-NOT: __ubsan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp -fopenmp-targets=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OMP-CPU
// CHECK-OMP-CPU-NOT: ubsan_offload
// CHECK-OMP-CPU-NOT: __ubsan_offload_init

int main(void) { return 0; }
