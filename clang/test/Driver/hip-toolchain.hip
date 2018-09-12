// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa" 
// CHECK-SAME: "-aux-triple" "x86_64--linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: "-fcuda-is-device" {{.*}} "-o" [[A_BC:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64--linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: "-fcuda-is-device" {{.*}} "-o" [[B_BC:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK: [[LLVM_LINK:"*.llvm-link"]] [[A_BC]] [[B_BC]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV1:".*-gfx803-linked-.*bc"]]

// CHECK: [[OPT:".*opt"]] [[LINKED_BC_DEV1]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx803"
// CHECK-SAME: "-o" [[OPT_BC_DEV1:".*-gfx803-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_BC_DEV1]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj" "-mcpu=gfx803" "-o" [[OBJ_DEV1:".*-gfx803-.*o"]]

// CHECK: [[LLD: ".*lld"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV1:.*out]]" [[OBJ_DEV1]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" 
// CHECK-SAME: "-aux-triple" "x86_64--linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: "-fcuda-is-device" {{.*}} "-o" [[A_BC:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64--linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: "-fcuda-is-device" {{.*}} "-o" [[B_BC:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[LLVM_LINK]] [[A_BC]] [[B_BC]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV2:".*-gfx900-linked-.*bc"]]

// CHECK: [[OPT]] [[LINKED_BC_DEV2]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx900"
// CHECK-SAME: "-o" [[OPT_BC_DEV2:".*-gfx900-optimized.*bc"]]

// CHECK: [[LLC]] [[OPT_BC_DEV2]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj" "-mcpu=gfx900" "-o" [[OBJ_DEV2:".*-gfx900-.*o"]]

// CHECK: [[LLD]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV2:.*out]]" [[OBJ_DEV2]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64--linux-gnu" 
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa" "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64--linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa" "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets={{.*}},hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// CHECK-SAME: "-inputs={{.*}},[[IMG_DEV1]],[[IMG_DEV2]]" "-outputs=[[BUNDLE:.*o]]"

// CHECK: [[LD:".*ld.*"]] {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]]
// CHECK-SAME: {{.*}} "-T" "{{.*}}.lk"
