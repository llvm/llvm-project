// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -x hip --gpu-max-threads-per-block=1024 %s 2>&1 | FileCheck %s

// Check that there are commands for both host- and device-side compilations.
//
// CHECK: clang{{.*}}" "-cc1" {{.*}} "-fcuda-is-device"
// CHECK-SAME: "--gpu-max-threads-per-block=1024"

// RUN: %clang -### -nogpuinc -nogpulib -fgpu-allow-device-init \
// RUN:   %s 2>&1 | FileCheck -check-prefix=DEVINIT %s
// DEVINIT: clang{{.*}}" "-cc1" {{.*}}"-fgpu-allow-device-init"
// DEVINIT: clang{{.*}}" "-cc1" {{.*}}"-fgpu-allow-device-init"

// RUN: %clang -### -x hip -target x86_64-pc-windows-msvc -fms-extensions \
// RUN:   -mllvm -amdgpu-early-inline-all=true  %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MLLVM %s
// MLLVM-NOT: "-mllvm"{{.*}}"-amdgpu-early-inline-all=true"{{.*}}"-mllvm"{{.*}}"-amdgpu-early-inline-all=true"

// RUN: %clang -### -Xarch_device -g -nogpulib --cuda-gpu-arch=gfx900 \
// RUN:   -Xarch_device -fcf-protection=branch \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=DEV %s
// DEV: clang{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}" {{.*}} "-fcf-protection=branch"
// DEV: clang{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}" {{.*}} "-fcf-protection=branch"
// DEV-NOT: clang{{.*}} {{.*}} "-debug-info-kind={{.*}}"

// RUN: %clang -### -Xarch_host -g -nogpulib --cuda-gpu-arch=gfx900 \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=HOST %s
// HOST-NOT: clang{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}"
// HOST-NOT: clang{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}"
// HOST: clang{{.*}} "-debug-info-kind={{.*}}"

// RUN: %clang -### -nogpuinc -nogpulib -munsafe-fp-atomics \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=UNSAFE-FP-ATOMICS %s
// UNSAFE-FP-ATOMICS: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-munsafe-fp-atomics"

// RUN: %clang -### -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=DEFAULT-UNSAFE-FP-ATOMICS %s
// DEFAULT-UNSAFE-FP-ATOMICS-NOT: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-munsafe-fp-atomics"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib -fgpu-exclude-wrong-side-overloads \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=FIX-OVERLOAD %s
// FIX-OVERLOAD: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fgpu-exclude-wrong-side-overloads" "-fgpu-defer-diag"
// FIX-OVERLOAD: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-fgpu-exclude-wrong-side-overloads" "-fgpu-defer-diag"

// Check -mconstructor-aliases is not passed to device compilation.

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=CTA %s
// CTA: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-mconstructor-aliases"
// CTA-NOT: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mconstructor-aliases"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 -fgpu-inline-threshold=1000 %s 2>&1 | FileCheck -check-prefix=THRESH %s
// THRESH: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mllvm" "-inline-threshold=1000"
// THRESH-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-inline-threshold=1000"

// Check -foffload-lto=thin translated correctly.

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -foffload-lto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIPTHINLTO %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -fgpu-rdc -foffload-lto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIPTHINLTO %s

// Ensure we don't error about -fwhole-program-vtables for the non-device offload compile.
// HIPTHINLTO-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'
// HIPTHINLTO-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto-unit"
// HIPTHINLTO: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-flto=thin" "-flto-unit" {{.*}} "-fwhole-program-vtables"
// HIPTHINLTO-NOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto-unit"
// HIPTHINLTO: lld{{.*}}"-plugin-opt=mcpu=gfx906" "-plugin-opt=thinlto" "-plugin-opt=-force-import-all"

// Check that -flto=thin is handled correctly, particularly with -fwhole-program-vtables.
//
// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -flto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=THINLTO %s

// Ensure we don't error about -fwhole-program-vtables for the device offload compile. We should
// drop -fwhole-program-vtables for the device offload compile and pass it through for the
// non-device offload compile along with -flto=thin.
// THINLTO-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'
// THINLTO-NOT: clang{{.*}}" "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fwhole-program-vtables"
// THINLTO: clang{{.*}}" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto=thin" {{.*}} "-fwhole-program-vtables"
// THINLTO-NOT: clang{{.*}}" "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fwhole-program-vtables"
