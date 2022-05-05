// Marking this as XFAIL since the steps checked below are different in both amd-stg-open and main
// XFAIL: *
// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t
// RUN: rm -f %t/amdgpu_arch_gfx906
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_gfx906 %t/
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_gfx908_gfx908 %t/
// RUN: chmod +x %t/amdgpu_arch_gfx906
// RUN: chmod +x %t/amdgpu_arch_gfx908_gfx908

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fno-openmp-new-driver -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "[[GFX:gfx906]]"
// CHECK: llvm-link{{.*}}"-o" "{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc"
// CHECK: llc{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=[[GFX]]" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-{{.*}}.o"

// case when amdgpu_arch returns multiple gpus but of same arch
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fno-openmp-new-driver -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib --amdgpu-arch-tool=%t/amdgpu_arch_gfx908_gfx908 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MULTIPLE
// CHECK-MULTIPLE: "-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "[[GFX:gfx908]]"
// CHECK-MULTIPLE: llvm-link{{.*}}"-o" "{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc"
// CHECK-MULTIPLE: llc{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=[[GFX]]" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-system-arch-{{.*}}-[[GFX]]-{{.*}}.o"
