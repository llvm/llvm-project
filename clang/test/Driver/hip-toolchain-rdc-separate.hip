// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -c -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -fgpu-rdc -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fgpu-rdc" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" "[[A_BC1:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fgpu-rdc"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" "[[A_BC2:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: {{.*}} "-o" "[[A_OBJ_HOST:.*o]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets=hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900,host-x86_64-unknown-linux-gnu"
// CHECK-SAME: "-outputs=[[A_O:.*a.o]]" "-inputs=[[A_BC1]],[[A_BC2]],[[A_OBJ_HOST]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fgpu-rdc" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" "[[B_BC1:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fgpu-rdc"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" "[[B_BC2:.*bc]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: {{.*}} "-o" "[[B_OBJ_HOST:.*o]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets=hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900,host-x86_64-unknown-linux-gnu"
// CHECK-SAME: "-outputs=[[B_O:.*b.o]]" "-inputs=[[B_BC1]],[[B_BC2]],[[B_OBJ_HOST]]"

// RUN: touch %T/a.o
// RUN: touch %T/b.o
// RUN: %clang --hip-link -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -fgpu-rdc -nogpuinc \
// RUN:   %T/a.o %T/b.o \
// RUN: 2>&1 | FileCheck -check-prefix=LINK %s

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-inputs=[[A_O:.*a.o]]" "-outputs=[[A_OBJ_HOST:.*o]],{{.*o}},{{.*o}}"
// LINK: "-unbundle"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-inputs=[[B_O:.*b.o]]" "-outputs=[[B_OBJ_HOST:.*o]],{{.*o}},{{.*o}}"
// LINK: "-unbundle"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-inputs=[[A_O]]" "-outputs={{.*o}},[[A_BC1:.*o]],[[A_BC2:.*o]]"
// LINK: "-unbundle"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-inputs=[[B_O]]" "-outputs={{.*o}},[[B_BC1:.*o]],[[B_BC2:.*o]]"
// LINK: "-unbundle"

// LINK-NOT: "*.llvm-link"
// LINK-NOT: ".*opt"
// LINK-NOT: ".*llc"
// LINK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// LINK: "-plugin-opt=mcpu=gfx803"
// LINK-SAME: "-o" "[[IMG_DEV1:.*.out]]" "[[A_BC1]]" "[[B_BC1]]"

// LINK-NOT: "*.llvm-link"
// LINK-NOT: ".*opt"
// LINK-NOT: ".*llc"
// LINK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// LINK: "-plugin-opt=mcpu=gfx900"
// LINK-SAME: "-o" "[[IMG_DEV2:.*.out]]" "[[A_BC2]]" "[[B_BC2]]"

// LINK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// LINK-SAME: "-targets={{.*}},hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// LINK-SAME: "-inputs={{.*}},[[IMG_DEV1]],[[IMG_DEV2]]" "-outputs=[[BUNDLE:.*hipfb]]"

// LINK: {{".*llvm-mc.*"}} "-triple" "amdgcn-amd-amdhsa" "-o"
// LINK-SAME: "[[OBJBUNDLE:.*o]]" "{{.*}}.mcin" "--filetype=obj"

// LINK: [[LD:".*ld.*"]] {{.*}} "-o" "a.out" {{.*}} "[[A_OBJ_HOST]]"
// LINK-SAME: "[[B_OBJ_HOST]]" "[[OBJBUNDLE]]"
