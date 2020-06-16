// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -mllvm -amdgpu-function-calls=0 \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-mllvm" "-amdgpu-function-calls=0" {{.*}}

// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-mllvm" "-amdgpu-function-calls=0" {{.*}}

// CHECK-NOT: {{".*opt"}}
// CHECK-NOT: {{".*llc"}}
