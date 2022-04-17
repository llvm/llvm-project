// Checks that cuda compilation does the right thing when passed -march.
// (Specifically, we want to pass it to host compilation, but not to device
// compilation or ptxas!)
//
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c \
// RUN: -march=haswell %s 2>&1 | FileCheck %s
// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c \
// RUN: -march=haswell --cuda-gpu-arch=sm_35 %s 2>&1 | FileCheck %s

// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK-SAME: "-triple" "nvptx
// CHECK-SAME: "-target-cpu" "sm_35"

// CHECK: ptxas
// CHECK-SAME: "--gpu-name" "sm_35"

// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK-SAME: "-target-cpu" "haswell"
