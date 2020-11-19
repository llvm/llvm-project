// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,DEFAULT %s

// RUN: %clang -### -O0 \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O0 %s

// RUN: %clang -### -O1 \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O1 %s

// RUN: %clang -### -O2 \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O2 %s

// RUN: %clang -### -O3 \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O3 %s

// RUN: %clang -### -Os \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Os %s

// RUN: %clang -### -Oz \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Oz %s

// RUN: %clang -### -Og \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Og %s

// ALL: "{{.*}}clang{{.*}}" "-cc1" "-mllvm" "--amdhsa-code-object-version=4" "-triple" "amdgcn-amd-amdhsa"
// DEFAULT-NOT: "-O{{.}}"
// O0-SAME: "-O0"
// O1-SAME: "-O1"
// O2-SAME: "-O2"
// O3-SAME: "-O3"
// Os-SAME: "-Os"
// Oz-SAME: "-Oz"
// Og-SAME: "-Og"

// ALL-NOT: "{{.*}}opt"

// ALL-NOT: "{{.*}}llc"

// ALL: "{{.*}}lld{{.*}}" {{.*}} "-plugin-opt=mcpu=gfx900"
// DEFAULT-NOT: "-plugin-opt=O{{.*}}"
// O0-SAME: "-plugin-opt=O0"
// O1-SAME: "-plugin-opt=O1"
// O2-SAME: "-plugin-opt=O2"
// O3-SAME: "-plugin-opt=O3"
// Os-SAME: "-plugin-opt=O2"
// Oz-SAME: "-plugin-opt=O2"
// Og-SAME: "-plugin-opt=O1"

// ALL: "{{.*}}clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// DEFAULT-NOT: "-O{{.}}"
// O0-SAME: "-O0"
// O1-SAME: "-O1"
// O2-SAME: "-O2"
// O3-SAME: "-O3"
// Os-SAME: "-Os"
// Oz-SAME: "-Oz"
// Og-SAME: "-Og"
