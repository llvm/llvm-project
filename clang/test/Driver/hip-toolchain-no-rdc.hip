// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK %s

//
// Compile device code in a.cu to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: "-fcuda-is-device" "-fvisibility" "hidden"
// CHECK-SAME: {{.*}} "-o" [[A_BC_803:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[LLVM_LINK:"*.llvm-link"]] [[A_BC_803]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV_A_803:".*-gfx803-linked-.*bc"]]

// CHECK: [[OPT:".*opt"]] [[LINKED_BC_DEV_A_803]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx803"
// CHECK-SAME: "-o" [[OPT_BC_DEV_A_803:".*-gfx803-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_BC_DEV_A_803]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj"
// CHECK-SAME: "-mattr=-code-object-v3"
// CHECK-SAME: "-mcpu=gfx803" "-o" [[OBJ_DEV_A_803:".*-gfx803-.*o"]]

// CHECK: [[LLD: ".*lld"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_A_803:.*out]]" [[OBJ_DEV_A_803]]

//
// Compile device code in a.cu to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: "-fcuda-is-device" "-fvisibility" "hidden"
// CHECK-SAME: {{.*}} "-o" [[A_BC_900:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[LLVM_LINK:"*.llvm-link"]] [[A_BC_900]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV_A_900:".*-gfx900-linked-.*bc"]]

// CHECK: [[OPT:".*opt"]] [[LINKED_BC_DEV_A_900]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx900"
// CHECK-SAME: "-o" [[OPT_BC_DEV_A_900:".*-gfx900-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_BC_DEV_A_900]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj"
// CHECK-SAME: "-mattr=-code-object-v3"
// CHECK-SAME: "-mcpu=gfx900" "-o" [[OBJ_DEV_A_900:".*-gfx900-.*o"]]

// CHECK: [[LLD: ".*lld"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_A_900:.*out]]" [[OBJ_DEV_A_900]]

//
// Bundle and embed device code in host object for a.cu.
//

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets={{.*}},hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// CHECK-SAME: "-inputs={{.*}},[[IMG_DEV_A_803]],[[IMG_DEV_A_900]]" "-outputs=[[BUNDLE_A:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa" "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]
// CHECK-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_A]]"

//
// Compile device code in b.hip to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: "-fcuda-is-device" "-fvisibility" "hidden"
// CHECK-SAME: {{.*}} "-o" [[B_BC_803:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK: [[LLVM_LINK:"*.llvm-link"]] [[B_BC_803]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV_B_803:".*-gfx803-linked-.*bc"]]

// CHECK: [[OPT:".*opt"]] [[LINKED_BC_DEV_B_803]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx803"
// CHECK-SAME: "-o" [[OPT_BC_DEV_B_803:".*-gfx803-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_BC_DEV_B_803]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj"
// CHECK-SAME: "-mattr=-code-object-v3"
// CHECK-SAME: "-mcpu=gfx803" "-o" [[OBJ_DEV_B_803:".*-gfx803-.*o"]]

// CHECK: [[LLD: ".*lld"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_B_803:.*out]]" [[OBJ_DEV_B_803]]

//
// Compile device code in b.hip to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: "-fcuda-is-device" "-fvisibility" "hidden"
// CHECK-SAME: {{.*}} "-o" [[B_BC_900:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK: [[LLVM_LINK:"*.llvm-link"]] [[B_BC_900]]
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-o" [[LINKED_BC_DEV_B_900:".*-gfx900-linked-.*bc"]]

// CHECK: [[OPT:".*opt"]] [[LINKED_BC_DEV_B_900]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx900"
// CHECK-SAME: "-o" [[OPT_BC_DEV_B_900:".*-gfx900-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_BC_DEV_B_900]] "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-filetype=obj"
// CHECK-SAME: "-mattr=-code-object-v3"
// CHECK-SAME: "-mcpu=gfx900" "-o" [[OBJ_DEV_B_900:".*-gfx900-.*o"]]

// CHECK: [[LLD: ".*lld"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_B_900:.*out]]" [[OBJ_DEV_B_900]]

//
// Bundle and embed device code in host object for b.hip.
//

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-targets={{.*}},hip-amdgcn-amd-amdhsa-gfx803,hip-amdgcn-amd-amdhsa-gfx900"
// CHECK-SAME: "-inputs={{.*}},[[IMG_DEV_B_803]],[[IMG_DEV_B_900]]" "-outputs=[[BUNDLE_A:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa" "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]
// CHECK-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_A]]"

//
// Link host objects.
//

// CHECK: [[LD:".*ld.*"]] {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]]
// CHECK-NOT: "-T" "{{.*}}.lk"
