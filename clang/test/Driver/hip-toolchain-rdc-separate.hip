// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: rm -rf %t && mkdir %t
// RUN: %clang -c -### --target=x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[A_BC1:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[A_BC2:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[A_OBJ_HOST:.*o]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets=hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900,host-x86_64-unknown-linux-gnu"
// CHECK-SAME: "-output=[[A_O:.*a.o]]" "-input=[[A_BC1]]" "-input=[[A_BC2]]" "-input=[[A_OBJ_HOST]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[B_BC1:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[B_BC2:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" "[[B_OBJ_HOST:.*o]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets=hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900,host-x86_64-unknown-linux-gnu"
// CHECK-SAME: "-output=[[B_O:.*b.o]]" "-input=[[B_BC1]]" "-input=[[B_BC2]]" "-input=[[B_OBJ_HOST]]"

// RUN: touch %t/a.o %t/b.o
// RUN: %clang --hip-link -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %t/a.o %t/b.o \
// RUN: 2>&1 | FileCheck -check-prefixes=LINK,LINK-HOST-UNBUNDLE,LLD-TMP,LINK-BUNDLE,LINK-EMBED %s

// RUN: %clang --hip-link -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %t/a.o %t/b.o --cuda-device-only \
// RUN: 2>&1 | FileCheck -check-prefixes=LINK,LLD-TMP,LINK-BUNDLE,LINK-NOEMBED %s

// RUN: %clang --hip-link -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %t/a.o %t/b.o --cuda-device-only --no-gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=LINK,LLD-FIN,LINK-NOBUNDLE,LINK-NOEMBED %s

// LINK-HOST-UNBUNDLE: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-HOST-UNBUNDLE-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-HOST-UNBUNDLE-SAME: "-input=[[A_O:.*a.o]]" "-output=[[A_OBJ_HOST:.*o]]" "-output={{.*o}}" "-output={{.*o}}"
// LINK-HOST-UNBUNDLE: "-unbundle" "-allow-missing-bundles"

// LINK-HOST-UNBUNDLE: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-HOST-UNBUNDLE-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-HOST-UNBUNDLE-SAME: "-input=[[B_O:.*b.o]]" "-output=[[B_OBJ_HOST:.*o]]" "-output={{.*o}}" "-output={{.*o}}"
// LINK-HOST-UNBUNDLE: "-unbundle" "-allow-missing-bundles"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-input=[[A_O:.*a.o]]" "-output={{.*o}}" "-output=[[A_BC1:.*o]]" "-output=[[A_BC2:.*o]]"
// LINK-SAME: "-unbundle" "-allow-missing-bundles"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-input=[[B_O:.*b.o]]" "-output={{.*o}}" "-output=[[B_BC1:.*o]]" "-output=[[B_BC2:.*o]]"
// LINK-SAME: "-unbundle" "-allow-missing-bundles"

// LINK-NOT: "*.llvm-link"
// LINK-NOT: ".*opt"
// LINK-NOT: ".*llc"
// LINK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// LINK-SAME: "-plugin-opt=mcpu=gfx803"
// LINK-SAME: "--whole-archive"
// LLD-TMP-SAME: "-o" "[[IMG_DEV1:.*.out]]"
// LLD-FIN-SAME: "-o" "[[IMG_DEV1:a.out-.*gfx803]]"
// LINK-SAME "[[A_BC1]]" "[[B_BC1]]"
// LINK-SAME: "--no-whole-archive"

// LINK-NOT: "*.llvm-link"
// LINK-NOT: ".*opt"
// LINK-NOT: ".*llc"
// LINK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// LINK-SAME: "-plugin-opt=mcpu=gfx900"
// LINK-SAME: "--whole-archive"
// LLD-TMP-SAME: "-o" "[[IMG_DEV2:.*.out]]"
// LLD-FIN-SAME: "-o" "[[IMG_DEV1:a.out-.*gfx900]]"
// LINK-SAME "[[A_BC2]]" "[[B_BC2]]"
// LINK-SAME: "--no-whole-archive"

// LINK-BUNDLE: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-BUNDLE-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// LINK-BUNDLE-SAME: "-input={{.*}}" "-input=[[IMG_DEV1]]" "-input=[[IMG_DEV2]]" "-output=[[BUNDLE:.*]]"
// LINK-NOBUNDLE-NOT: {{".*clang-offload-bundler"}} "-type=o"

// LINK-EMBED: {{".*llvm-mc.*"}} "-o" "[[OBJBUNDLE:.*o]]" "{{.*}}.mcin" "--filetype=obj"
// LINK-NOEMBED-NOT: {{".*llvm-mc.*"}} "-o"

// LINK-EMBED: [[LD:".*ld.*"]] {{.*}} "-o" "a.out" {{.*}} "[[A_OBJ_HOST]]"
// LINK-EMBED-SAME: "[[B_OBJ_HOST]]" "[[OBJBUNDLE]]"
// LINK-NOEMBED-NOT: {{".*ld.*"}} {{.*}} "-o" "a.out"
