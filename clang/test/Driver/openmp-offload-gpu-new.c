///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: clang{{.*}}"-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "sm_52"
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.*]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (nvptx64-nvidia-cuda)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (nvptx64-nvidia-cuda)" {8}, ir
// CHECK-PHASES: 10: backend, {9}, assembler, (host-openmp)
// CHECK-PHASES: 11: assembler, {10}, object, (host-openmp)
// CHECK-PHASES: 12: clang-linker-wrapper, {11}, image, (host-openmp)

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.*]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_OBJ:.*]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[DEVICE_OBJ]]"], output: "[[HOST_OBJ:.*]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: clang{{.*}}"-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"{{.*}}"-emit-llvm"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvida-cuda -march=sm_70 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-new-nvptx-test.bc \
// RUN:          -no-canonical-prefixes -nogpulib %s -o openmp-offload-gpu 2>&1 \
// RUN:   | FileCheck -check-prefix=DRIVER_EMBEDDING %s

// DRIVER_EMBEDDING: -fembed-offload-object=[[CUBIN:.*\.cubin]],openmp,nvptx64-nvidia-cuda,sm_70
