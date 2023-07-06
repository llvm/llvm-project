// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -x hip --gpu-max-threads-per-block=1024 %s 2>&1 | FileCheck %s

// Check that there are commands for both host- and device-side compilations.
//
// CHECK: "-cc1" {{.*}} "-fcuda-is-device"
// CHECK-SAME: "--gpu-max-threads-per-block=1024"

// RUN: %clang -### -nogpuinc -nogpulib -fgpu-allow-device-init \
// RUN:   %s 2>&1 | FileCheck -check-prefix=DEVINIT %s
// DEVINIT: "-cc1" {{.*}}"-fgpu-allow-device-init"
// DEVINIT: "-cc1" {{.*}}"-fgpu-allow-device-init"

// Check -fgpu-default-stream=per-thread.
// RUN: %clang -### -nogpuinc -nogpulib -fgpu-default-stream=per-thread \
// RUN:   %s -save-temps 2>&1 | FileCheck -check-prefix=PTH %s
// PTH: "-cc1"{{.*}} "-E" {{.*}}"-fgpu-default-stream=per-thread"
// PTH: "-cc1"{{.*}} "-fgpu-default-stream=per-thread" {{.*}}"-x" "hip-cpp-output"
// PTH: "-cc1"{{.*}} "-E" {{.*}}"-fgpu-default-stream=per-thread"
// PTH: "-cc1"{{.*}} "-fgpu-default-stream=per-thread" {{.*}}"-x" "hip-cpp-output"

// Check -mprintf-kind=hostcall
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -mprintf-kind=hostcall  %s -save-temps 2>&1 | FileCheck -check-prefix=HOSTC %s
// HOSTC: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-mprintf-kind=hostcall" "-Werror=format-invalid-specifier"{{.*}}"-E" {{.*}}
// HOSTC: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}} "-mprintf-kind=hostcall" "-Werror=format-invalid-specifier" {{.*}}"-x" "hip-cpp-output"
// HOSTC: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}} "-mprintf-kind=hostcall" "-Werror=format-invalid-specifier" {{.*}}"-x" "ir"
// HOSTC: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-E" {{.*}}
// HOSTC: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "ir"

// Check -mprintf-kind=buffered
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -mprintf-kind=buffered  %s -save-temps 2>&1 | FileCheck -check-prefix=BUFF %s
// BUFF: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-mprintf-kind=buffered" "-Werror=format-invalid-specifier"{{.*}}"-E" {{.*}}
// BUFF: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}} "-mprintf-kind=buffered" "-Werror=format-invalid-specifier" {{.*}}"-x" "hip-cpp-output"
// BUFF: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}} "-mprintf-kind=buffered" "-Werror=format-invalid-specifier" {{.*}}"-x" "ir"
// BUFF: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-E" {{.*}}
// BUFF: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "ir"

// RUN: %clang -### -x hip --target=x86_64-pc-windows-msvc -fms-extensions \
// RUN:   -mllvm -amdgpu-early-inline-all=true  %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MLLVM %s
// MLLVM-NOT: "-mllvm"{{.*}}"-amdgpu-early-inline-all=true"{{.*}}"-mllvm"{{.*}}"-amdgpu-early-inline-all=true"

// RUN: %clang -### -Xarch_device -g -nogpulib --cuda-gpu-arch=gfx900 \
// RUN:   -Xarch_device -fcf-protection=branch -Xarch_device -mllvm=--inline-threshold=100 \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=DEV %s
// DEV: "-cc1"{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}" {{.*}} "-fcf-protection=branch" {{.*}}"-mllvm" "--inline-threshold=100"
// DEV: "-cc1"{{.*}} "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}" {{.*}} "-fcf-protection=branch" {{.*}}"-mllvm" "--inline-threshold=100"
// DEV-NOT: clang{{.*}} {{.*}} "-debug-info-kind={{.*}}"

// RUN: %clang -### -Xarch_host -g -nogpulib --cuda-gpu-arch=gfx900 \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=HOST %s
// HOST-NOT: "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}"
// HOST-NOT: "-fcuda-is-device" {{.*}} "-debug-info-kind={{.*}}"
// HOST: clang{{.*}} "-debug-info-kind={{.*}}"

// RUN: %clang -### -nogpuinc -nogpulib -munsafe-fp-atomics \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=UNSAFE-FP-ATOMICS %s
// UNSAFE-FP-ATOMICS: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-munsafe-fp-atomics"

// RUN: %clang -### -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=DEFAULT-UNSAFE-FP-ATOMICS %s
// DEFAULT-UNSAFE-FP-ATOMICS-NOT: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-munsafe-fp-atomics"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib -fgpu-exclude-wrong-side-overloads \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=FIX-OVERLOAD %s
// FIX-OVERLOAD: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fgpu-exclude-wrong-side-overloads" "-fgpu-defer-diag"
// FIX-OVERLOAD: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-fgpu-exclude-wrong-side-overloads" "-fgpu-defer-diag"

// Check -mconstructor-aliases is not passed to device compilation.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=CTA %s
// CTA: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-mconstructor-aliases"
// CTA-NOT: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mconstructor-aliases"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 -fgpu-inline-threshold=1000 %s 2>&1 | FileCheck -check-prefix=THRESH %s
// THRESH: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mllvm" "-inline-threshold=1000"
// THRESH-NOT: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-inline-threshold=1000"

// Check -foffload-lto=thin translated correctly.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -foffload-lto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIPTHINLTO %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -fgpu-rdc -foffload-lto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIPTHINLTO %s

// Ensure we don't error about -fwhole-program-vtables for the non-device offload compile.
// HIPTHINLTO-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'
// HIPTHINLTO-NOT: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto-unit"
// HIPTHINLTO: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-flto=thin" "-flto-unit" {{.*}} "-fwhole-program-vtables"
// HIPTHINLTO-NOT: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto-unit"
// HIPTHINLTO: lld{{.*}}"-plugin-opt=mcpu=gfx906" "-plugin-opt=thinlto" "-plugin-opt=-force-import-all"

// Check that -flto=thin is handled correctly, particularly with -fwhole-program-vtables.
//
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -flto=thin -fwhole-program-vtables %s 2>&1 \
// RUN:   | FileCheck -check-prefix=THINLTO %s

// Ensure we don't error about -fwhole-program-vtables for the device offload compile. We should
// drop -fwhole-program-vtables for the device offload compile and pass it through for the
// non-device offload compile along with -flto=thin.
// THINLTO-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'
// THINLTO-NOT: "-cc1"{{.*}}" "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fwhole-program-vtables"
// THINLTO: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-flto=thin" {{.*}} "-fwhole-program-vtables"
// THINLTO-NOT: "-cc1"{{.*}}" "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fwhole-program-vtables"

// Check -fopenmp=libomp is allowed with HIP but -fopenmp-targets= is not allowed.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 -fopenmp=libomp %s 2>&1 | FileCheck -check-prefix=OMP %s
// OMP-NOT: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fopenmp"
// OMP: "-cc1"{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-fopenmp"

// RUN: not %clang --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 -fopenmp=libomp -fopenmp-targets=amdgcn %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OMPTGT %s
// OMPTGT: unsupported option '-fopenmp-targets=' for language mode 'HIP'

// Check -Xoffload-linker option is passed to lld.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -fgpu-rdc -Xoffload-linker --build-id=md5 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFL-LINK %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906 -Xoffload-linker --build-id=md5 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFL-LINK %s
// OFL-LINK: lld{{.*}}"--build-id=md5"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 -fhip-kernel-arg-name %s 2>&1 \
// RUN:   | FileCheck -check-prefix=KAN %s
// KAN: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fhip-kernel-arg-name"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=KANNEG %s
// KANNEG-NOT: "-fhip-kernel-arg-name"

// RUN: %clang -### -nogpuinc -nogpulib -mno-amdgpu-ieee -mamdgpu-ieee \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefix=IEEE-ON %s
// IEEE-ON-NOT: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-m{{(no-)?}}amdgpu-ieee"

// RUN: %clang -### -nogpuinc -nogpulib -mamdgpu-ieee -mno-amdgpu-ieee -ffast-math \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=IEEE-OFF %s
// IEEE-OFF: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mno-amdgpu-ieee"

// RUN: %clang -### -nogpuinc -nogpulib -mamdgpu-ieee -mno-amdgpu-ieee -ffast-math \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=IEEE-OFF-NEG %s
// IEEE-OFF-NEG-NOT: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mamdgpu-ieee"

// Check -fno-hip-fp32-correctly-rounded-divide-sqrt is passed to -cc1 but
// (default) -fhip-fp32-correctly-rounded-divide-sqrt is not.

// RUN: %clang -### -nogpuinc -nogpulib -fno-hip-fp32-correctly-rounded-divide-sqrt \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=NOCRDS %s
// NOCRDS: clang{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fno-hip-fp32-correctly-rounded-divide-sqrt"

// RUN: %clang -### -nogpuinc -nogpulib \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=CRDS %s
// CRDS-NOT: "-f{{(no-)?}}hip-fp32-correctly-rounded-divide-sqrt"

// RUN: %clang -### -nogpuinc -nogpulib -fhip-fp32-correctly-rounded-divide-sqrt \
// RUN:   --cuda-gpu-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=CRDS %s
// CRDS-NOT: "-f{{(no-)?}}hip-fp32-correctly-rounded-divide-sqrt"
