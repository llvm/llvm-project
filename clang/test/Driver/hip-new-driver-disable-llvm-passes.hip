// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -nogpuinc -nogpulib \
// RUN:   -O3 -c -x hip %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -nogpuinc -nogpulib -fgpu-rdc \
// RUN:   -O3 -c -x hip %s 2>&1 | FileCheck -check-prefix=RDC %s
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -nogpuinc -nogpulib -foffload-lto \
// RUN:   -O3 -c -x hip %s 2>&1 | FileCheck -check-prefix=LTO %s
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -nogpuinc -nogpulib \
// RUN:   -O3 --cuda-device-only -emit-llvm -c -x hip %s 2>&1 \
// RUN:   | FileCheck -check-prefix=USER-BC %s
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -nogpuinc -nogpulib \
// RUN:   -O3 --cuda-device-only -emit-llvm -S -x hip %s 2>&1 \
// RUN:   | FileCheck -check-prefix=USER-IR %s

// DEFAULT: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// DEFAULT-SAME: "-emit-llvm-bc"
// DEFAULT-SAME: "-disable-llvm-passes"
// DEFAULT: "{{.*}}llvm-offload-binary"
// DEFAULT: "{{.*}}clang-linker-wrapper"
// DEFAULT-SAME: "--no-lto"

// RDC: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// RDC-SAME: "-emit-llvm-bc"
// RDC-SAME: "-flto=full"
// RDC-NOT: "-disable-llvm-passes"

// LTO: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// LTO-SAME: "-emit-llvm-bc"
// LTO-SAME: "-flto=full"
// LTO-NOT: "-disable-llvm-passes"
// LTO: "{{.*}}clang-linker-wrapper"

// USER-BC: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// USER-BC-SAME: "-emit-llvm-bc"
// USER-BC-NOT: "-disable-llvm-passes"

// USER-IR: "-cc1" "-triple" "amdgcn-amd-amdhsa"
// USER-IR-SAME: "-emit-llvm"
// USER-IR-NOT: "-disable-llvm-passes"
