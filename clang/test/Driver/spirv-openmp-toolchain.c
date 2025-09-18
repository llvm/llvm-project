// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=spirv64-intel \
// RUN:        --libomptarget-spirv-bc-path=%t/ -nogpulib %s 2>&1 \
// RUN: | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "spirv64-intel" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-o" "{{.*}}.o"
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}} "-o" "a.out"

// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=spirv64-intel %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-PHASES %s

// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (spirv64-intel)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "device-openmp (spirv64-intel)" {8}, object
// CHECK-PHASES: 10: clang-offload-packager, {9}, image, (device-openmp)
// CHECK-PHASES: 11: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {10}, ir
// CHECK-PHASES: 12: backend, {11}, assembler, (host-openmp)
// CHECK-PHASES: 13: assembler, {12}, object, (host-openmp)
// CHECK-PHASES: 14: clang-linker-wrapper, {13}, image, (host-openmp)

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=spirv64-intel -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=spirv64-intel -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS

// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "spirv64-intel" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_SPV:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_SPV]]"], output: "[[DEVICE_IMAGE:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[DEVICE_IMAGE]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -save-temps -fopenmp=libomp -fopenmp-targets=spirv64-intel -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS-TEMPS
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -save-temps -fopenmp=libomp -fopenmp-targets=spirv64-intel %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS-TEMPS
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_PP:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_PP]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS-TEMPS: "spirv64-intel" - "clang", inputs: ["[[INPUT]]"], output: "[[DEVICE_PP:.+]]"
// CHECK-BINDINGS-TEMPS: "spirv64-intel" - "clang", inputs: ["[[DEVICE_PP]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS-TEMPS: "spirv64-intel" - "clang", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_ASM:.+]]"
// CHECK-BINDINGS-TEMPS: "spirv64-intel" - "SPIR-V::Assembler", inputs: ["[[DEVICE_ASM]]"], output: "[[DEVICE_SPV:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_SPV]]"], output: "[[DEVICE_IMAGE:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[DEVICE_IMAGE]]"], output: "[[HOST_ASM:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang::as", inputs: ["[[HOST_ASM]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp=libomp -fopenmp-targets=spirv64-intel -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: "-cc1" "-triple" "spirv64-intel"{{.*}}"-emit-llvm-bc"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=spirv64-intel \
// RUN: --sysroot=%S/Inputs/spirv-openmp/ %s 2>&1 | FileCheck --check-prefix=CHECK-GPULIB %s
// CHECK-GPULIB: "-cc1" "-triple" "spirv64-intel"{{.*}}"-mlink-builtin-bitcode" "{{.*}}libomptarget-spirv.bc"

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=spirv64-intel \
// RUN:        --libomptarget-spirv-bc-path=%t/ -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-OFFLOAD-ARCH-ERROR
// CHECK-OFFLOAD-ARCH-ERROR: error: failed to deduce triple for target architecture 'spirv64-intel'; specify the triple using '-fopenmp-targets' and '-Xopenmp-target' instead

// RUN: %clang -mllvm --spirv-ext=+SPV_INTEL_function_pointers -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=spirv64-intel \
// RUN:        -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-LINKER-ARG
// CHECK-LINKER-ARG: clang-linker-wrapper
// CHECK-LINKER-ARG-NOT: --device-linker=spirv64
// CHECK-LINKER-ARG-NOT: -mllvm
// CHECK-LINKER-ARG-NOT: --spirv-ext=+SPV_INTEL_function_pointers
// CHECK-LINKER-ARG: --linker-path
