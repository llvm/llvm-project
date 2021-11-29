// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

//
// RUN:   %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   %s -save-temps 2>&1 | FileCheck %s

// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-save-temps=cwd"{{.*}}"-x" "c"{{.*}}
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-save-temps=cwd"{{.*}}" "-o" "[[HOSTASM:.*.s]]" "-x" "ir"{{.*}}
// CHECK: clang{{.*}}"-cc1as" "-triple" "x86_64-unknown-linux-gnu" "-filetype" "obj"{{.*}}"-o" "[[HOSTOBJ:.*.o]]" "[[HOSTASM]]"

// compilation for offload target 1 : gfx906
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-save-temps=cwd"{{.*}}"-target-cpu" "gfx906"{{.*}}"-fopenmp-is-device"{{.*}}"-o" "{{.*}}.i" "-x" "c"{{.*}}.c
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-save-temps=cwd"{{.*}}"-target-cpu" "gfx906"{{.*}}"-fopenmp-is-device"{{.*}}"-o" "{{.*}}.bc" "-x" "cpp-output"{{.*}}.i
// CHECK: clang-build-select-link"{{.*}}openmp-offload-multi-save-temps-{{.*}}.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-select.bc"
// CHECK: llvm-link"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-select.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-linked.bc"
// CHECK: opt"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-linked.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-o"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-optimized.bc"
// CHECK: llc{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906-optimized.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-filetype=obj"{{.*}}"-o"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906.o"
// CHECK: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o" "[[GFX906OUT:.*.out]]" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx906.o"

// compilation for offload target 2 : gfx908
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-save-temps=cwd"{{.*}}"-target-cpu" "gfx908"{{.*}}"-fopenmp-is-device"{{.*}}"-o" "{{.*}}.i" "-x" "c"{{.*}}.c
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-save-temps=cwd"{{.*}}"-target-cpu" "gfx908"{{.*}}"-fopenmp-is-device"{{.*}}"-o" "{{.*}}.bc" "-x" "cpp-output"{{.*}}.i
// CHECK: clang-build-select-link"{{.*}}openmp-offload-multi-save-temps-{{.*}}.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-select.bc"
// CHECK: llvm-link"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-select.bc"{{.*}}"-o" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-linked.bc"
// CHECK: opt"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-linked.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx908" "-o"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-optimized.bc"
// CHECK: llc{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908-optimized.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx908" "-filetype=obj"{{.*}}"-o"{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908.o"
// CHECK: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o" "[[GFX908OUT:.*.out]]" "{{.*}}openmp-offload-multi-save-temps-{{.*}}-gfx908.o" "-plugin-opt=mcpu=gfx908"

// Combining device images for offload targets
// CHECK: clang-offload-wrapper"{{.*}}" "-o" "[[COMBINEDIR:.*.bc]]" "--offload-arch=gfx906" "[[GFX906OUT]]" "--offload-arch=gfx908" "[[GFX908OUT]]"

// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-S" "-save-temps=cwd"{{.*}}"-fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa"{{.*}}"-o" "[[COMBINEDASM:.*.s]]" "-x" "ir" "[[COMBINEDIR]]"
// CHECK: clang{{.*}}"-cc1as" "-triple" "x86_64-unknown-linux-gnu" "-filetype" "obj"{{.*}}"-o" "[[COMBINEDOBJ:.*.o]]" "[[COMBINEDASM]]"
// CHECK: ld"{{.*}}" "-o" "a.out{{.*}}[[HOSTOBJ]]" "[[COMBINEDOBJ]]{{.*}}" "-lomp{{.*}}-lomptarget"
