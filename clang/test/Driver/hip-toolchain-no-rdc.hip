// RUN: rm -rf %t && mkdir %t
// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -nogpuinc -c \
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
// RUN:   --offload-arch=amdgcnspirv --offload-arch=gfx900 -no-use-spirv-backend \
// RUN:   %s -nogpuinc -nogpulib \
// RUN: 2>&1 | FileCheck -check-prefixes=AMDGCNSPIRV-NEW %s

//
// Compile device code in a.cu to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgpu8.03-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// NEW-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_803:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

//
// Compile device code in a.cu to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_900:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

//
// Bundle and embed device code in host object for a.cu.
//

// NEW: [[PACKAGER:".*llvm-offload-binary"]] "-o" "[[PACKAGE_A:.*.out]]"
// NEW-SAME: "--image=file=[[OBJ_DEV_A_803]],triple=amdgpu8.03-amd-amdhsa,arch=gfx803,kind=hip"
// NEW-SAME: "--image=file=[[OBJ_DEV_A_900]],triple=amdgpu9.00-amd-amdhsa,arch=gfx900,kind=hip"

// NEW: [[WRAPPER:".*clang-linker-wrapper]]"
// NEW-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
// NEW-SAME: "--host-triple=x86_64-unknown-linux-gnu"
// NEW-SAME: "--emit-fatbin-only"
// NEW-SAME: "-o" "[[HIPFB_A:.*.hipfb]]" "[[PACKAGE_A]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgpu-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// NEW-SAME: {{.*}} "-foffload-include-binary" "[[HIPFB_A]]"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

//
// Compile device code in b.hip to code object for gfx803.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgpu8.03-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_B_803:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

//
// Compile device code in b.hip to code object for gfx900.
//

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-{{(obj|llvm-bc)}}"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_B_900:.*(o|bc)]]" "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK-NOT: {{".*llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

//
// Bundle and embed device code in host object for b.hip.
//

// NEW: [[PACKAGER:".*llvm-offload-binary"]] "-o" "[[PACKAGE_B:.*.out]]"
// NEW-SAME: "--image=file=[[OBJ_DEV_B_803]],triple=amdgpu8.03-amd-amdhsa,arch=gfx803,kind=hip"
// NEW-SAME: "--image=file=[[OBJ_DEV_B_900]],triple=amdgpu9.00-amd-amdhsa,arch=gfx900,kind=hip"

// NEW: [[WRAPPER:".*clang-linker-wrapper]]"
// NEW-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
// NEW-SAME: "--emit-fatbin-only"
// NEW-SAME: "-o" "[[HIPFB_B:.*.hipfb]]" "[[PACKAGE_B]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "amdgpu-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// NEW-SAME: {{.*}} "-foffload-include-binary" "[[HIPFB_B]]"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

//
// Link host objects.
//

// LKONLY-NOT: {{".*/llvm-link"}}
// LKONLY-NOT: {{".*/opt"}}
// LKONLY-NOT: {{".*/llc"}}
// LKONLY: [[LD:".*ld.*"]] {{.*}} "{{.*/a.o}}" "{{.*/b.o}}"
// LKONLY-NOT: "-T" "{{.*}}.lk"

//
// Check mixed AMDGCNSPIRV and concrete GPU arch.
//

// AMDGCNSPIRV-NEW: "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}}"-emit-llvm-bc" {{.*}} "-o" "[[AMDGCNSPV_BC:[^"]*]]"

// Check verbose printing with the new driver.
// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx908 -v %s 2>&1 | FileCheck %s --check-prefix=VERBOSE
// VERBOSE: clang-linker-wrapper
// VERBOSE-NOT: --device-compiler=amdgpu{{.*}}-amd-amdhsa=-v {{.*}}-o {{.*}}.hipfb
// VERBOSE-NOT: --wrapper-verbose {{.*}}-o {{.*}}.hipfb
