// REQUIRES: x86-registered-target
// DESIRES: amdgpu-registered-target
// REQUIRES: working-afar-ubuntu
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --no-opaque-offload-linker --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          --offload-arch=gfx906 --no-opaque-offload-linker --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: clang{{.*}}"-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "gfx906"{{.*}}"-mlink-builtin-bitcode"
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-obj"
// CHECK: clang-linker-wrapper{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
// RUN:   -march=gfx906 %s 2>&1 | FileCheck --check-prefix=CHECK-PHASES %s
// phases
// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp, gfx906)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp, gfx906)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp, gfx906)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa:gfx906)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, ir, (device-openmp, gfx906)
// CHECK-PHASES: 8: offload, "device-openmp (amdgcn-amd-amdhsa:gfx906)" {7}, ir
// CHECK-PHASES: 9: clang-offload-packager, {8}, image, (device-openmp)
// CHECK-PHASES: 10: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {9}, ir
// CHECK-PHASES: 11: backend, {10}, assembler, (host-openmp)
// CHECK-PHASES: 12: assembler, {11}, object, (host-openmp)
// CHECK-PHASES: 13: clang-linker-wrapper, {12}, image, (host-openmp)

// handling of --libomptarget-amdgpu-bc-path
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIBOMPTARGET
// CHECK-LIBOMPTARGET: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOGPULIB
// CHECK-NOGPULIB-NOT: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_BC]]"], output: "[[BINARY:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --no-opaque-offload-linker -lm --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIB-DEVICE-NEW
// CHECK-LIB-DEVICE-NEW: {{.*}}"-target-cpu" "gfx803"{{.*}}ocml.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"
