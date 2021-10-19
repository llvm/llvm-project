// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Output bundled code objects for combined compilation.
// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck %s

// CHECK: {{.*}}clang-offload-bundler{{.*}}"-outputs=hip-output-file-name.o"

// Check -E default output is "-" (stdout).
// If there are multiple preprocessor expansion outputs clang-offload-bundler
// is used to bundle the final output.

// Output bundled PPE for one GPU for mixed compliation.
// RUN: %clang -### -E -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// Output unbundled PPE for one GPU for device only compilation.
// RUN: %clang -### -E --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// Output bundled PPE for two GPUs for mixed compilation.
// RUN: %clang -### -E -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// Output bundled PPE for two GPUs for mixed compilation with -save-temps.
// RUN: %clang -### -E -save-temps -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// Output unbundled PPE for two GPUs for device only compilation.
// RUN: %clang -### -E --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// Output bundled PPE for two GPUs for device only compilation with --gpu-bundle-output.
// RUN: %clang -### -E --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=DASH %s

// Output unbundled PPE for two GPUs for device only compilation with --no-gpu-bundle-output.
// RUN: %clang -### -E --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --no-gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// Output unbundled PPE for host only compilation.
// RUN: %clang -### -E --cuda-host-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-DASH %s

// DASH-NOT: {{.*}}clang{{.*}}"-o" "-"
// DASH: {{.*}}clang-offload-bundler{{.*}}"-outputs=-"
// CLANG-DASH: {{.*}}clang{{.*}}"-o" "-"
// CLANG-DASH-NOT: {{.*}}clang-offload-bundler{{.*}}"-outputs=-"

// Check -E with -o.

// Output bundled PPE for two GPUs for mixed compilation.
// RUN: %clang -### -E -o test.cui -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=OUT %s

// Output bundled PPE for two GPUs for mixed compilation.
// RUN: %clang -### -E -o test.cui -save-temps -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=OUT %s

// Output bundled PPE for two GPUs for device only compilation with --gpu-bundle-output.
// RUN: %clang -### -E -o test.cui --cuda-device-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 --gpu-bundle-output %s \
// RUN: 2>&1 | FileCheck -check-prefixes=OUT %s

// Output unbundled PPE for two GPUs for device only compilation.
// RUN: %clang -### -E -o test.cui --cuda-host-only -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck -check-prefixes=CLANG-OUT %s

// OUT-NOT: {{.*}}clang{{.*}}"-o" "test.cui"
// OUT: {{.*}}clang-offload-bundler{{.*}}"-outputs=test.cui"
// CLANG-OUT: {{.*}}clang{{.*}}"-o" "test.cui"
// CLANG-OUT-NOT: {{.*}}clang-offload-bundler{{.*}}"-outputs=test.cui"
