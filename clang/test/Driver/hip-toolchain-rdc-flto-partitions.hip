// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --flto-partitions=42 \
// RUN:   --no-offload-new-driver --emit-static-lib -nogpulib \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefix=FIXED-PARTS

// FIXED-PARTS-NOT: "*.llvm-link"
// FIXED-PARTS-NOT: ".*opt"
// FIXED-PARTS-NOT: ".*llc"
// FIXED-PARTS: [[LLD: ".*lld.*"]] {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// FIXED-PARTS-SAME: "-plugin-opt=mcpu=gfx803"
// FIXED-PARTS-SAME: "--lto-partitions=42"
// FIXED-PARTS-SAME: "-o" "{{.*out}}" "{{.*bc}}"

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --flto-partitions=a \
// RUN:   --no-offload-new-driver --emit-static-lib -nogpulib \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefix=LTO_PARTS_INV0

// LTO_PARTS_INV0: clang: error: invalid integral value 'a' in '--flto-partitions=a'

// RUN: not %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --flto-partitions=0 \
// RUN:   --no-offload-new-driver --emit-static-lib -nogpulib \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefix=LTO_PARTS_INV1

// LTO_PARTS_INV1: clang: error: invalid integral value '0' in '--flto-partitions=0'
