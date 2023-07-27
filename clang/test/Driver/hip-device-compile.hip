// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// If -emit-llvm and/or -S is used in device only compilation,
// the output should not be bundled, except --gpu-bundle-output
// is specified.

// Output unbundled bitcode.
// RUN: %clang -c -emit-llvm --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.bc -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,BC,NBUN %s

// Output bundled bitcode.
// RUN: %clang -c -emit-llvm --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.bc -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,BCBUN %s

// Output unbundled LLVM IR.
// RUN: %clang -c -S -emit-llvm --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.ll -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LL,NBUN %s

// Output bundled LLVM IR.
// RUN: %clang -c -S -emit-llvm --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.ll -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LLBUN %s

// Output unbundled assembly.
// RUN: %clang -c -S --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.s -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,ASM,NBUN %s

// Output relocatable.
// RUN: %clang -c --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.o -x hip --cuda-gpu-arch=gfx900 -fhip-emit-relocatable \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,NBUN,RELOC %s

// Output bundled assembly.
// RUN: %clang -c -S --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.s -x hip --cuda-gpu-arch=gfx900 --no-gpu-bundle-output \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,ASMBUN %s

// CHECK: {{".*clang.*"}} "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// BC-SAME: "-emit-llvm-bc"
// LL-SAME: "-emit-llvm"
// ASM-NOT: "-emit-llvm"
// CHECK-SAME: "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: {{".*lib1.bc"}}
// CHECK-SAME: "-target-cpu" "gfx900"
// BC-SAME: "-o" "a.bc"
// BCBUN-SAME: "-o" "{{.*}}.bc"
// LL-SAME: "-o" "a.ll"
// LLBUN-SAME: "-o" "{{.*}}.ll"
// ASM-SAME: "-o" "a.s"
// ASMBUN-SAME: "-o" "{{.*}}.s"
// RELOC-SAME: "-o" "a.o"
// CHECK-SAME: {{".*a.cu"}}

// CHECK-NOT: {{"*.llvm-link"}}
// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}
// CHECK-NOT: {{".*lld.*"}}
// NBUN-NOT: {{".*clang-offload-bundler"}}
// BCBUN: {{".*clang-offload-bundler"}}{{.*}}"-output=a.bc"
// LLBUN: {{".*clang-offload-bundler"}}{{.*}}"-output=a.ll"
// ASMBUN: {{".*clang-offload-bundler"}}{{.*}}"-output=a.s"
// CHECK-NOT: {{".*ld.*"}}{{.*}}"-o"

// If neither -emit-llvm nor -S is used in device only compilation,
// the output should be bundled except --no-gpu-bundle-output is
// specified.

// Output bundled code objects.
// RUN: %clang -c --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.o -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=OBJ,OBJ-BUN %s

// Output unbundled code objects.
// RUN: %clang -c --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.o -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu --no-gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=OBJ,OBJ-UBUN %s

// Output bundled code objects.
// RUN: %clang --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.o -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck -check-prefixes=OBJ,OBJ-BUN %s

// Output unbundled code objects.
// RUN: %clang --cuda-device-only -### --target=x86_64-linux-gnu \
// RUN:   -o a.o -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu --no-gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=OBJ,OBJ-UBUN %s

// OBJ: {{"*.clang.*"}} {{.*}} "-emit-obj"
// OBJ-NOT: {{"*.llvm-link"}}
// OBJ-NOT: {{".*opt"}}
// OBJ-NOT: {{".*llc"}}
// OBJ-BUN: {{".*lld.*"}}{{.*}}"-o" "{{.*}}.o"
// OBJ-UBUN: {{".*lld.*"}}{{.*}}"-o" "a.o"
// OBJ-BUN: {{".*clang-offload-bundler"}}{{.*}}"-output=a.o"
// OBJ-UBUN-NOT: {{".*clang-offload-bundler"}}
