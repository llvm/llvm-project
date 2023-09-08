// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// Check that we enable LTO-mode properly with '-fopenmp-target-jit' and that it
// still enabled LTO-mode if `-fno-offload-lto` is on.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-phases -fopenmp=libomp \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=PHASES-JIT %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-phases -fopenmp=libomp \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -foffload-lto -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=PHASES-JIT %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-phases -fopenmp=libomp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=PHASES-JIT %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-phases -fopenmp=libomp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -foffload-lto -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=PHASES-JIT %s
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-phases -fopenmp=libomp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fno-offload-lto -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=PHASES-JIT %s
//
//      PHASES-JIT: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// PHASES-JIT-NEXT: 1: preprocessor, {0}, cpp-output, (host-openmp)
// PHASES-JIT-NEXT: 2: compiler, {1}, ir, (host-openmp)
// PHASES-JIT-NEXT: 3: input, "[[INPUT]]", c, (device-openmp)
// PHASES-JIT-NEXT: 4: preprocessor, {3}, cpp-output, (device-openmp)
// PHASES-JIT-NEXT: 5: compiler, {4}, ir, (device-openmp)
// PHASES-JIT-NEXT: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp ([[TARGET:.+]])" {5}, ir
// PHASES-JIT-NEXT: 7: backend, {6}, lto-bc, (device-openmp)
// PHASES-JIT-NEXT: 8: offload, "device-openmp ([[TARGET]])" {7}, lto-bc
// PHASES-JIT-NEXT: 9: clang-offload-packager, {8}, image, (device-openmp)
// PHASES-JIT-NEXT: 10: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {9}, ir
// PHASES-JIT-NEXT: 11: backend, {10}, assembler, (host-openmp)
// PHASES-JIT-NEXT: 12: assembler, {11}, object, (host-openmp)
// PHASES-JIT-NEXT: 13: clang-linker-wrapper, {12}, image, (host-openmp)

// Check that we add the `--embed-bitcode` flag to the linker wrapper.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_52 \
// RUN:   -fopenmp-target-jit %s 2>&1 | FileCheck -check-prefix=LINKER %s
// LINKER: clang-linker-wrapper"{{.*}}"--embed-bitcode"

// Check for incompatible combinations

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fno-offload-lto \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-LTO %s
// NO-LTO: error: the combination of '-fno-offload-lto' and '-fopenmp-target-jit' is incompatible

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -foffload-lto=thin \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-target-jit %s 2>&1 \
// RUN: | FileCheck -check-prefix=THIN-LTO %s
// THIN-LTO: error: the combination of '-foffload-lto=' and '-fopenmp-target-jit' is incompatible
