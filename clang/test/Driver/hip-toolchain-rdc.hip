// Check -flto-partitions

// RUN: %clang -### -fgpu-rdc --offload-arch=gfx90a -nogpulib -nogpuinc \
// RUN:   -L. -foffload-lto %s 2>&1 | FileCheck -check-prefix=LTO_DEFAULT_NEW %s
// LTO_DEFAULT_NEW: clang-linker-wrapper
// LTO_DEFAULT_NEW-NOT: "--device-compiler=amdgpu-amd-amdhsa=-flto-partitions=
