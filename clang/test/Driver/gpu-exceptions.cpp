// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXC
// RUN: %clang -### --target=nvptx64-nvidia-cuda -march=sm_80 -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXC
// RUN: %clang -### --target=spirv64-- -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXC

// Check that the default can still be overridden.
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib -fexceptions %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXC

// Check that offloading languages still use their own handling.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx90a -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OPENMP

// NOEXC-NOT: "-fcxx-exceptions"
// NOEXC-NOT: "-fexceptions"

// EXC: "-fcxx-exceptions"
// EXC: "-fexceptions"

// OPENMP: "-cc1"{{.*}}"-triple" "amdgpu9.0a-amd-amdhsa"
// OPENMP-SAME: "-fcxx-exceptions"

int main() { return 0; }
