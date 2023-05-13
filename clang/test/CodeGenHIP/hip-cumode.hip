// REQUIRES: amdgpu-registered-target

// RUN: %clang -S -o - --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=NOWGP %s
// RUN: %clang -S -o - --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib -mcumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=NOWGP %s
// RUN: %clang -S -o - --offload-arch=gfx906 --cuda-device-only -nogpuinc -nogpulib -mno-cumode \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=NOWGP,WARN-CUMODE %s
// RUN: %clang -S -o - --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-OFF %s
// RUN: %clang -S -o - --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib -mcumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-ON %s
// RUN: %clang -S -o - --offload-arch=gfx1030 --cuda-device-only -nogpuinc -nogpulib -mno-cumode \
// RUN:   %s 2>&1 | FileCheck --check-prefix=CUMODE-OFF %s
// WARN-CUMODE: warning: ignoring '-mno-cumode' option as it is not currently supported for processor 'gfx906' [-Woption-ignored]
// NOWGP-NOT: .amdhsa_workgroup_processor_mode
// CUMODE-ON: .amdhsa_workgroup_processor_mode 0
// CUMODE-OFF: .amdhsa_workgroup_processor_mode 1

__attribute__((global)) void kernel() {}
