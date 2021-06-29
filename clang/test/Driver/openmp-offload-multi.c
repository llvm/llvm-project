// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "c"{{.*}}
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-o" "[[HOSTOBJ:.*.o]]" "-x" "ir"{{.*}}

// compilation for offload target 1 : gfx906
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-target-cpu" "gfx906" "-fcuda-is-device"{{.*}}"-o" "{{.*}}.bc" "-x" "c"{{.*}}.c
// CHECK: llvm-link"{{.*}}openmp-offload-multi-{{.*}}-gfx906-select-{{.*}}.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-{{.*}}-gfx906-linked-{{.*}}.bc"
// CHECK: opt"{{.*}}openmp-offload-multi-{{.*}}-gfx906-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-o"{{.*}}openmp-offload-multi-{{.*}}-gfx906-optimized-{{.*}}.bc"
// CHECK: llc{{.*}}openmp-offload-multi-{{.*}}-gfx906-optimized-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-filetype=obj"{{.*}}"-o"{{.*}}openmp-offload-multi-{{.*}}-gfx906-{{.*}}.o"
// CHECK: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o" "[[GFX906OUT:.*.out]]" "{{.*}}openmp-offload-multi-{{.*}}-gfx906-{{.*}}.o"

// compilation for offload target 2 : gfx908
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-target-cpu" "gfx908" "-fcuda-is-device"{{.*}}"-o" "{{.*}}.bc" "-x" "c"{{.*}}.c
// CHECK: llvm-link"{{.*}}openmp-offload-multi-{{.*}}-gfx908-select-{{.*}}.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-{{.*}}-gfx908-linked-{{.*}}.bc"
// CHECK: opt"{{.*}}openmp-offload-multi-{{.*}}-gfx908-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx908" "-o"{{.*}}openmp-offload-multi-{{.*}}-gfx908-optimized-{{.*}}.bc"
// CHECK: llc{{.*}}openmp-offload-multi-{{.*}}-gfx908-optimized-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx908" "-filetype=obj"{{.*}}"-o"{{.*}}openmp-offload-multi-{{.*}}-gfx908-{{.*}}.o"
// CHECK: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o" "[[GFX908OUT:.*.out]]" "{{.*}}openmp-offload-multi-{{.*}}-gfx908-{{.*}}.o"

// Combining device images for offload targets
// CHECK: clang-offload-wrapper"{{.*}}" "-o" "[[COMBINEDIR:.*.bc]]" "--offload-arch=gfx906" "[[GFX906OUT]]" "--offload-arch=gfx908" "[[GFX908OUT]]"

// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa"{{.*}}"-o" "[[COMBINEDOBJ:.*.o]]" "-x" "ir" "[[COMBINEDIR]]"
// CHECK: ld.lld"{{.*}}" "-o" "a.out{{.*}}[[HOSTOBJ]]" "[[COMBINEDOBJ]]{{.*}}" "-lomp{{.*}}-lomptarget"