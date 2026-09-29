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
// CHECK-HIP-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload-preinit.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "-u" "__csan_offload_init"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-DEV \
// RUN:       --implicit-check-not=csan_offload-preinit
// CHECK-SHARED-DEV-DAG: "-u" "__csan_offload_init"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=concurrency \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED \
// RUN:       --implicit-check-not=csan_offload-preinit \
// RUN:       --implicit-check-not=libclang_rt.csan.a
// CHECK-SHARED-DAG: "-u" "__csan_offload_init"
// CHECK-SHARED-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=concurrency -shared-libsan \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-RT \
// RUN:       --implicit-check-not=csan_offload.a \
// RUN:       --implicit-check-not=libclang_rt.csan.a
// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -shared-libsan -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_csan %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-RT \
// RUN:       --implicit-check-not=csan_offload.a \
// RUN:       --implicit-check-not=libclang_rt.csan.a
// CHECK-SHARED-RT: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.so"
// CHECK-SHARED-RT: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload-preinit.a" "--no-whole-archive"

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
