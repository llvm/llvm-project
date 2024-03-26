// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check -Xoffload-linker -mllvm=* options are passed
// to device lld linker only.
// -mllvm options are passed to clang and device lld linker.

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx803 --offload-arch=gfx900 \
// RUN:   -nogpuinc -nogpulib -mllvm -unroll-count=10 \
// RUN:   -Xoffload-linker -mllvm=-inline-threshold=100 \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx803 --offload-arch=gfx900 \
// RUN:   -nogpuinc -nogpulib -mllvm -unroll-count=10 \
// RUN:   -Xoffload-linker -mllvm=-inline-threshold=100 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NEG %s

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx803 --offload-arch=gfx900 \
// RUN:   -nogpuinc -nogpulib -fgpu-rdc -mllvm -unroll-count=10 \
// RUN:   -Xoffload-linker -mllvm=-inline-threshold=100 \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx803 --offload-arch=gfx900 \
// RUN:   -nogpuinc -nogpulib -fgpu-rdc -mllvm -unroll-count=10 \
// RUN:   -Xoffload-linker -mllvm=-inline-threshold=100 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NEG %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-mllvm" "-unroll-count=10" {{.*}}

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-mllvm" "-unroll-count=10" {{.*}}

// NEG-NOT: {{".*opt"}}
// NEG-NOT: {{".*llc"}}
// NEG-NOT: "-m" "elf_x86_64"{{.*}} "-plugin-opt=-unroll-count=10"
// NEG-NOT: "-m" "elf_x86_64"{{.*}} "-plugin-opt=-inline-threshold=100"
