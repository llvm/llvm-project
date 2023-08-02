// RUN: %clang -target x86_64-unknown-linux-gnu --sysroot %S/Inputs/basic_cross_linux_tree %s \
// RUN: -fno-amd-opt -flto -O3 -### 2>&1 | FileCheck --check-prefix=CHECK-LTO-OPEN  %s
// CHECK-LTO-OPEN-NOT: "{{.*}}../alt/bin/clang-{{.*}}"
// CHECK-LTO-OPEN-NOT: "{{.*}}../alt/bin/ld.lld"

// RUN: %clang -target x86_64-unknown-linux-gnu --sysroot %S/Inputs/basic_cross_linux_tree %s \
// RUN: -fno-amd-opt -O3 -### 2>&1 | FileCheck --check-prefix=CHECK-OPEN  %s
// CHECK-OPEN-NOT: "{{.*}}../alt/bin/clang-{{.*}}"
// CHECK-OPEN-NOT: "{{.*}}../alt/bin/ld.lld"

// RUN: not %clang -target x86_64-unknown-linux-gnu --sysroot %S/Inputs/basic_cross_linux_tree %s \
// RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN: -fno-amd-opt -flto -O3 -### 2>&1 | FileCheck --check-prefix=CHECK-OMP-LTO-OPEN  %s
// CHECK-OMP-LTO-OPEN-NOT: "{{.*}}../alt/bin/clang-{{.*}}"
// CHECK-OMP-LTO-OPEN-NOT: "{{.*}}../alt/bin/ld.lld"

// RUN: not %clang -target x86_64-unknown-linux-gnu --sysroot %S/Inputs/basic_cross_linux_tree %s \
// RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN: -fno-amd-opt -O3 -### 2>&1 | FileCheck --check-prefix=CHECK-OMP-OPEN  %s
// CHECK-OMP-OPEN-NOT: "{{.*}}../alt/bin/clang-{{.*}}"
// CHECK-OMP-OPEN-NOT: "{{.*}}../alt/bin/ld.lld"

// RUN: %clang -famd-opt -O3 -### %s  2>&1 | FileCheck --check-prefix=CHECK-ALT-MISS  %s
// CHECK-ALT-MISS: warning: The [AMD] proprietary optimization compiler installation was not found
// RUN: %clang -fveclib=AMDLIBM -O3 -### %s  2>&1 | FileCheck --check-prefix=CHECK-VECLIB  %s
// CHECK-VECLIB: warning: The [AMD] proprietary optimization compiler installation was not found
