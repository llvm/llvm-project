// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mxnack 2>&1 | FileCheck %s -check-prefix=XNACK
// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-xnack 2>&1 | FileCheck %s -check-prefix=NOXNACK

// XNACK: {{.*}}clang{{.*}}"-target-feature" "+xnack"
// XNACK: {{.*}}lld{{.*}}"-plugin-opt=-mattr=+xnack"
// NOXNACK: {{.*}}clang{{.*}}"-target-feature" "-xnack"
// NOXNACK: {{.*}}lld{{.*}}"-plugin-opt=-mattr=-xnack"


// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -msram-ecc 2>&1 | FileCheck %s -check-prefix=SRAM
// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-sram-ecc 2>&1 | FileCheck %s -check-prefix=NOSRAM

// SRAM: {{.*}}clang{{.*}}"-target-feature" "+sram-ecc"
// SRAM: {{.*}}lld{{.*}}"-plugin-opt=-mattr=+sram-ecc"
// NOSRAM: {{.*}}clang{{.*}}"-target-feature" "-sram-ecc"
// NOSRAM: {{.*}}lld{{.*}}"-plugin-opt=-mattr=-sram-ecc"


// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mxnack -msram-ecc \
// RUN:   2>&1 | FileCheck %s -check-prefix=ALL3
// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-xnack -mno-sram-ecc \
// RUN:   2>&1 | FileCheck %s -check-prefix=NOALL3

// ALL3: {{.*}}clang{{.*}}"-target-feature" "+xnack" "-target-feature" "+sram-ecc"
// ALL3: {{.*}}lld{{.*}}"-plugin-opt=-mattr=+xnack,+sram-ecc"
// NOALL3: {{.*}}clang{{.*}}"-target-feature" "-xnack" "-target-feature" "-sram-ecc"
// NOALL3: {{.*}}lld{{.*}}"-plugin-opt=-mattr=-xnack,-sram-ecc"

// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx1010 %s \
// RUN:   -mcumode -mcumode -mno-cumode -mwavefrontsize64 -mcumode \
// RUN:   -mwavefrontsize64 -mno-wavefrontsize64 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DUP
// DUP: {{.*}}clang{{.*}} "-target-feature" "-wavefrontsize16"
// DUP-SAME: "-target-feature" "+wavefrontsize32"
// DUP-SAME: "-target-feature" "-wavefrontsize64"
// DUP-SAME: "-target-feature" "+cumode"
// DUP: {{.*}}lld{{.*}} "-plugin-opt=-mattr=-wavefrontsize16,+wavefrontsize32,-wavefrontsize64,+cumode"
