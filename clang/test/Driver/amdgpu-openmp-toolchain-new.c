// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: clang{{.*}}"-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "gfx906"{{.*}}"-fcuda-is-device"{{.*}}"-mlink-builtin-bitcode" "{{.*}}libomptarget-amdgpu-gfx906.bc"
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.*]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa)" {8}, ir
// CHECK-PHASES: 10: backend, {9}, assembler, (host-openmp)
// CHECK-PHASES: 11: assembler, {10}, object, (host-openmp)
// CHECK-PHASES: 12: clang-linker-wrapper, {11}, image, (host-openmp)

// handling of --libomptarget-amdgpu-bc-path
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIBOMPTARGET
// CHECK-LIBOMPTARGET: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOGPULIB
// CHECK-NOGPULIB-NOT: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.*]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[DEVICE_BC]]"], output: "[[HOST_OBJ:.*]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -lm --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIB-DEVICE-NEW
// CHECK-LIB-DEVICE-NEW: {{.*}}clang-linker-wrapper{{.*}}-target-library=openmp-amdgcn-amd-amdhsa-gfx803={{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"
