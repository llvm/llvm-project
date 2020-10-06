// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck %s

// CHECK: {{.*}}clang-offload-bundler{{.*}}"-outputs=hip-output-file-name.o"

// Check -E default output is "-" (stdout).

// RUN: %clang -### -E -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// RUN: %clang -### -E -save-temps -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// RUN: %clang -### -E --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// RUN: %clang -### -E --cuda-host-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// DASH: {{.*}}clang-offload-bundler{{.*}}"-outputs=-"
// CLANG-DASH: {{.*}}clang{{.*}}"-o" "-"

// Check -E with -o.

// RUN: %clang -### -E -o test.cui -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=OUT %s

// RUN: %clang -### -E -o test.cui -save-temps -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=OUT %s

// RUN: %clang -### -E -o test.cui --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-OUT %s

// RUN: %clang -### -E -o test.cui --cuda-host-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-OUT %s

// OUT: {{.*}}clang-offload-bundler{{.*}}"-outputs=test.cui"
// CLANG-OUT: {{.*}}clang{{.*}}"-o" "test.cui"
