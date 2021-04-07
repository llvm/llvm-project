// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LINK %s

// RUN: %clang -### -target x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -nogpuinc -c \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK %s

// RUN: touch %T/a.o
// RUN: touch %T/b.o
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --hip-link --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -nogpuinc \
// RUN:   %T/a.o %T/b.o \
// RUN: 2>&1 | FileCheck -check-prefixes=LKONLY %s

//
// Compile device code in a.cu to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-mllvm" "--amdhsa-code-object-version={{[0-9]+}}" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" [[OBJ_DEV_A_803:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// CHECK: [[LLD: ".*lld.*"]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_A_803:.*out]]" [[OBJ_DEV_A_803]]

//
// Compile device code in a.cu to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-mllvm" "--amdhsa-code-object-version={{[0-9]+}}" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" [[OBJ_DEV_A_900:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// CHECK: [[LLD]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_A_900:.*out]]" [[OBJ_DEV_A_900]]

//
// Bundle and embed device code in host object for a.cu.
//

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-bundle-align=4096"
// CHECK-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// CHECK-SAME: "-inputs={{.*}},[[IMG_DEV_A_803]],[[IMG_DEV_A_900]]" "-outputs=[[BUNDLE_A:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_A]]"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

//
// Compile device code in b.hip to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-mllvm" "--amdhsa-code-object-version={{[0-9]+}}" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" [[OBJ_DEV_B_803:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// CHECK: [[LLD]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_B_803:.*out]]" [[OBJ_DEV_B_803]]

//
// Compile device code in b.hip to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-mllvm" "--amdhsa-code-object-version={{[0-9]+}}" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" [[OBJ_DEV_B_900:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// CHECK: [[LLD]] "-flavor" "gnu" "--no-undefined" "-shared"
// CHECK-SAME: "-o" "[[IMG_DEV_B_900:.*out]]" [[OBJ_DEV_B_900]]

//
// Bundle and embed device code in host object for b.hip.
//

// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-bundle-align=4096"
// CHECK-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// CHECK-SAME: "-inputs={{.*}},[[IMG_DEV_B_803]],[[IMG_DEV_B_900]]" "-outputs=[[BUNDLE_A:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_A]]"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

//
// Link host objects.
//

// LINK-NOT: {{".*/llvm-link"}}
// LINK-NOT: {{".*/opt"}}
// LINK-NOT: {{".*/llc"}}
// LINK: [[LD:".*ld.*"]] {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]]
// LINK-NOT: "-T" "{{.*}}.lk"

// LKONLY-NOT: {{".*/llvm-link"}}
// LKONLY-NOT: {{".*/opt"}}
// LKONLY-NOT: {{".*/llc"}}
// LKONLY: [[LD:".*ld.*"]] {{.*}} "{{.*/a.o}}" "{{.*/b.o}}"
// LKONLY-NOT: "-T" "{{.*}}.lk"
