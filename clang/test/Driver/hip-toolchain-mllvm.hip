// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -mllvm -amdgpu-function-calls=0 \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-mllvm" "-amdgpu-function-calls=0" {{.*}}

// CHECK: [[OPT:".*opt"]] {{".*-gfx803-linked.*bc"}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx803" "-amdgpu-function-calls=0"
// CHECK-SAME: "-o" [[OPT_803_BC:".*-gfx803-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_803_BC]]
// CHECK-SAME: "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-mcpu=gfx803"
// CHECK-SAME: "-filetype=obj"
// CHECK-SAME: "-amdgpu-function-calls=0" "-o" {{".*-gfx803-.*o"}}

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-mllvm" "-amdgpu-function-calls=0" {{.*}}

// CHECK: [[OPT]] {{".*-gfx900-linked.*bc"}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx900" "-amdgpu-function-calls=0"
// CHECK-SAME: "-o" [[OPT_900_BC:".*-gfx900-optimized.*bc"]]

// CHECK: [[LLC]] [[OPT_900_BC]]
// CHECK-SAME: "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-mcpu=gfx900"
// CHECk-SAME: "-filetype=obj"
// CHECK-SAME: "-amdgpu-function-calls=0" "-o" {{".*-gfx900-.*o"}}
