// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%S/Inputs/amdgpu-arch/amdgpu_arch_gfx906 %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "[[GFX:gfx906]]"
// CHECK: llvm-link{{.*}}"-o" "{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc"
// CHECK: llc{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=[[GFX]]" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-{{.*}}.o"

// case when amdgpu_arch returns multiple gpus but of same arch
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%S/Inputs/amdgpu-arch/amdgpu_arch_gfx908_gfx908 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MULTIPLE
// CHECK-MULTIPLE: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "[[GFX:gfx908]]"
// CHECK-MULTIPLE: llvm-link{{.*}}"-o" "{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc"
// CHECK-MULTIPLE: llc{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=[[GFX]]" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-{{.*}}.o"
