// RUN: %clang -### \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,DEFAULT %s

// RUN: %clang -### -O0 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O0 %s

// RUN: %clang -### -O1 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O1 %s

// RUN: %clang -### -O2 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O2 %s

// RUN: %clang -### -O3 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O3 %s

// RUN: %clang -### -Os \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Os %s

// RUN: %clang -### -Oz \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Oz %s

// RUN: %clang -### -Og \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,Og %s

// RUN: %clang -### -O0 \
// RUN:   -Xoffload-linker --lto-CGO2 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,O0-CGO2 %s

// ALL: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// DEFAULT-NOT: "-O{{.}}"
// O0-SAME: "-O0"
// O1-SAME: "-O1"
// O2-SAME: "-O2"
// O3-SAME: "-O3"
// Os-SAME: "-Os"
// Oz-SAME: "-Oz"
// Og-SAME: "-Og"
// O0-CGO2-SAME: "-O0"
// O0-CGO2-NOT: "--lto-CGO2"

// ALL: "{{.*}}clang-linker-wrapper{{.*}}" "--should-extract=gfx900"
// DEFAULT-NOT: "--device-compiler=amdgcn-amd-amdhsa=-O{{.*}}"
// O0-SAME: "--device-compiler=amdgcn-amd-amdhsa=-O0"
// O1-SAME: "--device-compiler=amdgcn-amd-amdhsa=-O1"
// O2-SAME: "--device-compiler=amdgcn-amd-amdhsa=-O2"
// O3-SAME: "--device-compiler=amdgcn-amd-amdhsa=-O3"
// Os-SAME: "--device-compiler=amdgcn-amd-amdhsa=-Os"
// Oz-SAME: "--device-compiler=amdgcn-amd-amdhsa=-Oz"
// Og-SAME: "--device-compiler=amdgcn-amd-amdhsa=-Og"
// O0-CGO2-SAME: "--device-compiler=amdgcn-amd-amdhsa=-O0"

// ALL: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// DEFAULT-NOT: "-O{{.}}"
// O0-SAME: "-O0"
// O1-SAME: "-O1"
// O2-SAME: "-O2"
// O3-SAME: "-O3"
// Os-SAME: "-Os"
// Oz-SAME: "-Oz"
// Og-SAME: "-Og"
// O0-CGO2-SAME: "-O0"
// O0-CGO2-NOT: "--lto-CGO2"
