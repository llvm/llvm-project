// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mxnack 2>&1 | FileCheck %s -check-prefix=XNACK
// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-xnack 2>&1 | FileCheck %s -check-prefix=NOXNACK

// XNACK: {{.*}}clang{{.*}}"-target-feature" "+xnack"
// NOXNACK: {{.*}}clang{{.*}}"-target-feature" "-xnack"


// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -msram-ecc 2>&1 | FileCheck %s -check-prefix=SRAM
// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-sram-ecc 2>&1 | FileCheck %s -check-prefix=NOSRAM

// SRAM: {{.*}}clang{{.*}}"-target-feature" "+sram-ecc"
// NOSRAM: {{.*}}clang{{.*}}"-target-feature" "-sram-ecc"


// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mxnack -msram-ecc \
// RUN:   2>&1 | FileCheck %s -check-prefix=ALL3
// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -mno-xnack -mno-sram-ecc \
// RUN:   2>&1 | FileCheck %s -check-prefix=NOALL3

// ALL3: {{.*}}clang{{.*}}"-target-feature" "+xnack" "-target-feature" "+sram-ecc"
// NOALL3: {{.*}}clang{{.*}}"-target-feature" "-xnack" "-target-feature" "-sram-ecc"
