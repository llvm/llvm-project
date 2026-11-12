// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=daddress -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=daddress"
// CHECK-OPENMP-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan-preinit.a" "--no-whole-archive"
// CHECK-OPENMP-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=daddress \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan-preinit.a" "--no-whole-archive"
// CHECK-XARCH-DEV-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=daddress \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=daddress -shared-libsan \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-RT \
// RUN:       --implicit-check-not=x86_64-unknown-linux-gnu{{.}}libclang_rt.dasan.a
// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=daddress \
// RUN:     -shared-libsan -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-RT \
// RUN:       --implicit-check-not=x86_64-unknown-linux-gnu{{.}}libclang_rt.dasan.a
// CHECK-SHARED-RT: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan.so"
// CHECK-SHARED-RT: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan-preinit.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=daddress -shared-libsan \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-DSO \
// RUN:       --implicit-check-not=dasan-preinit \
// RUN:       --implicit-check-not=x86_64-unknown-linux-gnu{{.}}libclang_rt.dasan.a
// CHECK-SHARED-DSO: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan.so"

// RUN: %clang -no-canonical-prefixes -### --target=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:     -nogpulib -fsanitize=daddress -Xlinker -lc \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE
//      CHECK-DEVICE: ld.lld
//  CHECK-DEVICE-NOT: dasan-preinit
// CHECK-DEVICE-SAME: "--whole-archive" "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.dasan.a" "--no-whole-archive"
// CHECK-DEVICE-SAME: "-lc"

// RUN: %clang -no-canonical-prefixes -### --target=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:     -nogpulib -fsanitize=daddress -shared-libsan -Xlinker -lc \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE --implicit-check-not=dasan.so

// RUN: %clang -no-canonical-prefixes -### -x hip --offload-arch=gfx908 \
// RUN:     --offload-device-only -fsanitize=daddress -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP
//      CHECK-HIP: lld
// CHECK-HIP-SAME: "--whole-archive"
// CHECK-HIP-SAME: "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.dasan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=daddress -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MISSING
// CHECK-MISSING-NOT: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=daddress"

int main(void) { return 0; }
