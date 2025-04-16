// RUN: rm -rf %t && mkdir %t
// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LINK,OLD %s

// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -nogpuinc -c \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,OLD %s

// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -nogpuinc --offload-new-driver -c \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,NEW %s

// RUN: touch %t/a.o %t/b.o
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --hip-link --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -nogpuinc \
// RUN:   %t/a.o %t/b.o \
// RUN: 2>&1 | FileCheck -check-prefixes=LKONLY %s

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=amdgcnspirv --offload-arch=gfx900 \
// RUN:   %s -nogpuinc -nogpulib \
// RUN: 2>&1 | FileCheck -check-prefixes=AMDGCNSPIRV %s

//
// Compile device code in a.cu to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// OLD-SAME: "-emit-obj"
// NEW-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_803:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// OLD: [[LLD: ".*lld.*"]] "-flavor" "gnu" "-m" "elf64_amdgpu" "--no-undefined" "-shared"
// OLD-SAME: "-o" "[[IMG_DEV_A_803:.*out]]" "[[OBJ_DEV_A_803]]"

//
// Compile device code in a.cu to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_900:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// OLD: [[LLD]] "-flavor" "gnu" "-m" "elf64_amdgpu" "--no-undefined" "-shared"
// OLD-SAME: "-o" "[[IMG_DEV_A_900:.*out]]" "[[OBJ_DEV_A_900]]"

//
// Bundle and embed device code in host object for a.cu.
//

// OLD: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// OLD-SAME: "-bundle-align=4096"
// OLD-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// OLD-SAME: "-input={{.*}}" "-input=[[IMG_DEV_A_803]]" "-input=[[IMG_DEV_A_900]]" "-output=[[BUNDLE_A:.*hipfb]]"

// NEW: [[PACKAGER:".*clang-offload-packager"]] "-o" "[[PACKAGE_A:.*.out]]"
// NEW-SAME: "--image=file=[[OBJ_DEV_A_803]],triple=amdgcn-amd-amdhsa,arch=gfx803,kind=hip"
// NEW-SAME: "--image=file=[[OBJ_DEV_A_900]],triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// OLD-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_A]]"
// NEW-SAME: {{.*}} "-fembed-offload-object=[[PACKAGE_A]]"
// OLD-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// NEW-SAME: {{.*}} "-o" [[A_OBJ_HOST_TMP:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// NEW: [[WRAPPER:".*clang-linker-wrapper]]" {{.*}}"--host-triple=x86_64-unknown-linux-gnu"
// NEW:   "--linker-path={{.*}}" "-o" [[A_OBJ_HOST:".*o"]] [[A_OBJ_HOST_TMP]] "-r"

//
// Compile device code in b.hip to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_B_803:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// OLD: [[LLD]] "-flavor" "gnu" "-m" "elf64_amdgpu" "--no-undefined" "-shared"
// OLD-SAME: "-o" "[[IMG_DEV_B_803:.*out]]" "[[OBJ_DEV_B_803]]"

//
// Compile device code in b.hip to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_B_900:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// OLD: [[LLD]] "-flavor" "gnu" "-m" "elf64_amdgpu" "--no-undefined" "-shared"
// OLD-SAME: "-o" "[[IMG_DEV_B_900:.*out]]" "[[OBJ_DEV_B_900]]"

//
// Bundle and embed device code in host object for b.hip.
//

// OLD: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// OLD-SAME: "-bundle-align=4096"
// OLD-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// OLD-SAME: "-input={{.*}}" "-input=[[IMG_DEV_B_803]]" "-input=[[IMG_DEV_B_900]]" "-output=[[BUNDLE_B:.*hipfb]]"

// NEW: [[PACKAGER:".*clang-offload-packager"]] "-o" "[[PACKAGE_B:.*.out]]"
// NEW-SAME: "--image=file=[[OBJ_DEV_B_803]],triple=amdgcn-amd-amdhsa,arch=gfx803,kind=hip"
// NEW-SAME: "--image=file=[[OBJ_DEV_B_900]],triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// OLD-SAME: {{.*}} "-fcuda-include-gpubinary" "[[BUNDLE_B]]"
// NEW-SAME: {{.*}} "-fembed-offload-object=[[PACKAGE_B]]"
// OLD-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// NEW-SAME: {{.*}} "-o" [[B_OBJ_HOST_TMP:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// NEW: [[WRAPPER:".*clang-linker-wrapper]]" {{.*}}"--host-triple=x86_64-unknown-linux-gnu"
// NEW:   "--linker-path={{.*}}" "-o" [[B_OBJ_HOST:".*o"]] [[B_OBJ_HOST_TMP]] "-r"

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

//
// Check mixed AMDGCNSPIRV and concrete GPU arch.
//

// AMDGCNSPIRV: "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}}"-emit-llvm-bc" {{.*}} "-o" "[[AMDGCNSPV_BC:.*bc]]"
// AMDGCNSPIRV: {{".*llvm-link.*"}} "-o" "[[AMDGCNSPV_TMP:.*out]]" "[[AMDGCNSPV_BC]]"
// AMDGCNSPIRV: {{".*llvm-spirv.*"}} "--spirv-max-version=1.6" "--spirv-ext=+all" {{.*}} "[[AMDGCNSPV_TMP]]" {{.*}}"-o" "[[AMDGCNSPV_CO:.*out]]"
// AMDGCNSPIRV: "-cc1" "-triple" "amdgcn-amd-amdhsa" {{.*}}"-emit-obj" {{.*}}"-target-cpu" "gfx900"{{.*}} "-o" "[[GFX900_OBJ:.*o]]"
// AMDGCNSPIRV: {{".*lld.*"}} {{.*}}"-plugin-opt=mcpu=gfx900" {{.*}} "-o" "[[GFX900_CO:.*out]]" {{.*}}"[[GFX900_OBJ]]"
// AMDGCNSPIRV: {{".*clang-offload-bundler.*"}} "-type=o"
// AMDGCNSPIRV-SAME: "-targets={{.*}}hipv4-spirv64-amd-amdhsa--amdgcnspirv,hipv4-amdgcn-amd-amdhsa--gfx900"
// AMDGCNSPIRV-SAME: "-input=[[AMDGCNSPV_CO]]" "-input=[[GFX900_CO]]"
