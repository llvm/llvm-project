// REQUIRES: amdgpu-registered-target

// Check that -mamdgpu-fp32-recip-newton-raphson passes -mllvm flag to device cc1.

// RUN: %clang -### -x hip -nogpulib -nogpuinc --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 -mamdgpu-fp32-recip-newton-raphson %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ON %s

// ON: "-cc1" {{.*}} "-fcuda-is-device" {{.*}} "-mllvm" "-amdgpu-enable-fp32-recip-newton-raphson"

// Check that -mno-amdgpu-fp32-recip-newton-raphson does not pass the flag.

// RUN: %clang -### -x hip -nogpulib -nogpuinc --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 -mno-amdgpu-fp32-recip-newton-raphson %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFF %s

// OFF-NOT: "-amdgpu-enable-fp32-recip-newton-raphson"

// Check that -mno overrides -m (last flag wins).

// RUN: %clang -### -x hip -nogpulib -nogpuinc --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 -mamdgpu-fp32-recip-newton-raphson \
// RUN:   -mno-amdgpu-fp32-recip-newton-raphson %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFF %s

// Check default (off).

// RUN: %clang -### -x hip -nogpulib -nogpuinc --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFF %s
