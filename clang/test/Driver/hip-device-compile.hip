// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// If -emit-llvm and/or -S is used in device only compilation,
// the output should not be bundled.

// RUN: %clang -c -emit-llvm --cuda-device-only -### -target x86_64-linux-gnu \
// RUN:   -o a.bc -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,BC %s

// RUN: %clang -c -S -emit-llvm --cuda-device-only -### -target x86_64-linux-gnu \
// RUN:   -o a.ll -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LL %s

// RUN: %clang -c -S --cuda-device-only -### -target x86_64-linux-gnu \
// RUN:   -o a.s -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,ASM %s

// CHECK: {{".*clang.*"}} "-cc1" "-mllvm" "--amdhsa-code-object-version={{[0-9]+}}" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// BC-SAME: "-emit-llvm-bc"
// LL-SAME: "-emit-llvm"
// ASM-NOT: "-emit-llvm"
// CHECK-SAME: "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: {{".*lib1.bc"}}
// CHECK-SAME: "-target-cpu" "gfx900"
// BC-SAME: "-o" "a.bc"
// LL-SAME: "-o" "a.ll"
// ASM-SAME: "-o" "a.s"
// CHECK-SAME: {{".*a.cu"}}

// CHECK-NOT: {{"*.llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}
// CHECK-NOT: {{".*lld.*"}}
// CHECK-NOT: {{".*clang-offload-bundler"}}
// CHECK-NOT: {{".*ld.*"}}

// If neither -emit-llvm nor -S is used in device only compilation,
// the output should be bundled.

// RUN: %clang -c --cuda-device-only -### -target x86_64-linux-gnu \
// RUN:   -o a.s -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=BUNDLE %s

// RUN: %clang --cuda-device-only -### -target x86_64-linux-gnu \
// RUN:   -o a.s -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=BUNDLE %s

// BUNDLE: {{"*.clang.*"}} {{.*}} "-emit-obj"
// BUNDLE-NOT: {{"*.llvm-link"}}
// BUNDLE-NOT: {{".*opt"}}
// BUNDLE-NOT: {{".*llc"}}
// BUNDLE: {{".*lld.*"}}
// BUNDLE: {{".*clang-offload-bundler"}}

