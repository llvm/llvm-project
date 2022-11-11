// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          --offload-arch=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "gfx906"{{.*}}"-fcuda-is-device"{{.*}}"-mlink-builtin-bitcode" "{{.*}}libomptarget-amdgpu-gfx906.bc"
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "device-openmp (amdgcn-amd-amdhsa)" {8}, object
// CHECK-PHASES: 10: clang-offload-packager, {9}, image
// CHECK-PHASES: 11: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {10}, ir
// CHECK-PHASES: 12: backend, {11}, assembler, (host-openmp)
// CHECK-PHASES: 13: assembler, {12}, object, (host-openmp)
// CHECK-PHASES: 14: clang-linker-wrapper, {13}, image, (host-openmp)

// handling of --libomptarget-amdgpu-bc-path
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIBOMPTARGET
// CHECK-LIBOMPTARGET: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOGPULIB
// CHECK-NOGPULIB-NOT: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_BC]]"], output: "[[BINARY:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx803 \
// RUN:   --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s  2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-LIB-DEVICE
// CHECK-LIB-DEVICE: "-cc1" {{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx803 -nogpulib \
// RUN:   --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s  2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-LIB-DEVICE-NOGPULIB
// CHECK-LIB-DEVICE-NOGPULIB-NOT: "-cc1" {{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"
