///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

/// ###########################################################################

/// Check -Xopenmp-target uses one of the archs provided when several archs are used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target -march=sm_35 -Xopenmp-target -march=sm_60 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-ARCHS %s

// CHK-FOPENMP-TARGET-ARCHS: ptxas{{.*}}" "--gpu-name" "sm_60"

/// ###########################################################################

/// Check -Xopenmp-target -march=sm_35 works as expected when two triples are present.
// RUN:   %clang -### -fopenmp=libomp \
// RUN:          -fopenmp-targets=powerpc64le-ibm-linux-gnu,nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_35 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-COMPILATION %s

// CHK-FOPENMP-TARGET-COMPILATION: ptxas{{.*}}" "--gpu-name" "sm_35"

/// Check PTXAS is passed -c flag when offloading to an NVIDIA device using OpenMP.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-DEFAULT %s

// CHK-PTXAS-DEFAULT: ptxas{{.*}}" "-c"

/// ###########################################################################

/// PTXAS is passed -c flag by default when offloading to an NVIDIA device using OpenMP - disable it.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -fnoopenmp-relocatable-target \
// RUN:          -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-NORELO %s

// CHK-PTXAS-NORELO-NOT: ptxas{{.*}}" "-c"

/// ###########################################################################

/// PTXAS is passed -c flag by default when offloading to an NVIDIA device using OpenMP
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-relocatable-target \
// RUN:          -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-RELO %s

// CHK-PTXAS-RELO: ptxas{{.*}}" "-c"

/// ###########################################################################

/// Check that error is not thrown by toolchain when no cuda lib flag is used.
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 \
// RUN:   -nocudalib -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FLAG-NOLIBDEVICE %s

// CHK-FLAG-NOLIBDEVICE-NOT: error:{{.*}}sm_60

/// ###########################################################################

/// Check that error is not thrown by toolchain when no cuda lib device is found when using -S.
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -S -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 \
// RUN:   -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NOLIBDEVICE %s

// CHK-NOLIBDEVICE-NOT: error:{{.*}}sm_60

/// ###########################################################################

/// Check that the runtime bitcode library is part of the compile line.
/// Create a bogus bitcode library and specify it with libomptarget-nvptx-bc-path
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB %s

/// Specify the directory containing the bitcode lib, check clang picks the right one
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHK-BCLIB-DIR %s

/// Create a bogus bitcode library and find it with LIBRARY_PATH
// RUN:   env LIBRARY_PATH=%S/Inputs/libomptarget/subdir %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHK-ENV-BCLIB %s

// CHK-BCLIB: clang{{.*}}-triple{{.*}}nvptx64-nvidia-cuda{{.*}}-mlink-builtin-bitcode{{.*}}libomptarget-nvptx-test.bc
// CHK-BCLIB-DIR: clang{{.*}}-triple{{.*}}nvptx64-nvidia-cuda{{.*}}-mlink-builtin-bitcode{{.*}}libomptarget{{/|\\\\}}libomptarget-nvptx-sm_35.bc
// CHK-ENV-BCLIB: clang{{.*}}-triple{{.*}}nvptx64-nvidia-cuda{{.*}}-mlink-builtin-bitcode{{.*}}subdir{{/|\\\\}}libomptarget-nvptx-sm_35.bc
// CHK-BCLIB-NOT: {{error:|warning:}}

/// ###########################################################################

/// Check that the warning is thrown when the libomptarget bitcode library is not found.
/// Libomptarget requires sm_35 or newer so an sm_35 bitcode library should never exist.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB-WARN %s

// CHK-BCLIB-WARN: no library 'libomptarget-nvptx-sm_35.bc' found in the default clang lib directory or in LIBRARY_PATH; use '--libomptarget-nvptx-bc-path' to specify nvptx bitcode library

/// ###########################################################################

/// Check that the error is thrown when the libomptarget bitcode library does not exist.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --libomptarget-nvptx-bc-path=not-exist.bc \
// RUN:   -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB-ERROR %s

// CHK-BCLIB-ERROR: bitcode library 'not-exist.bc' does not exist

/// ###########################################################################

/// Check that the error is thrown when CUDA 9.1 or lower version is used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_35 --cuda-path=%S/Inputs/CUDA_90/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUDA-VERSION-ERROR %s

// CHK-CUDA-VERSION-ERROR: NVPTX target requires CUDA 9.2 or above; CUDA 9.0 detected

/// Check that debug info is emitted in dwarf-2
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O1 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g0 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb0 -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -gline-directives-only 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s

// DEBUG_DIRECTIVES-NOT: warning: debug
// NO_DEBUG-NOT: warning: debug
// NO_DEBUG: "-fopenmp-is-device"
// NO_DEBUG-NOT: "-debug-info-kind=
// NO_DEBUG: ptxas
// DEBUG_DIRECTIVES: "-triple" "nvptx64-nvidia-cuda"
// DEBUG_DIRECTIVES-SAME: "-debug-info-kind=line-directives-only"
// DEBUG_DIRECTIVES-SAME: "-fopenmp-is-device"
// DEBUG_DIRECTIVES: ptxas
// DEBUG_DIRECTIVES: "-lineinfo"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O0 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O0 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g2 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb2 -O0 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g3 -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb3 -O2 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -gline-tables-only 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb1 -O2 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s

// HAS_DEBUG-NOT: warning: debug
// HAS_DEBUG: "-triple" "nvptx64-nvidia-cuda"
// HAS_DEBUG-SAME: "-debug-info-kind={{constructor|line-tables-only}}"
// HAS_DEBUG-SAME: "-dwarf-version=2"
// HAS_DEBUG-SAME: "-fopenmp-is-device"
// HAS_DEBUG: ptxas
// HAS_DEBUG-SAME: "-g"
// HAS_DEBUG-SAME: "--dont-merge-basicblocks"
// HAS_DEBUG-SAME: "--return-at-end"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-mode -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-mode -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// CUDA_MODE: "-cc1"{{.*}}"-triple" "{{nvptx64-nvidia-cuda|amdgcn-amd-amdhsa}}"
// CUDA_MODE-SAME: "-fopenmp-cuda-mode"
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-mode -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-mode -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// NO_CUDA_MODE-NOT: "-{{fno-|f}}openmp-cuda-mode"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-teams-reduction-recs-num=2048 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_RED_RECS %s
// CUDA_RED_RECS: "-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"
// CUDA_RED_RECS-SAME: "-fopenmp-cuda-teams-reduction-recs-num=2048"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OPENMP_NVPTX_WRAPPERS %s
// OPENMP_NVPTX_WRAPPERS: "-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"
// OPENMP_NVPTX_WRAPPERS-SAME: "-internal-isystem" "{{.*}}openmp_wrappers"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -save-temps -ccc-print-bindings %s -o openmp-offload-gpu 2>&1 \
// RUN:   | FileCheck -check-prefix=SAVE_TEMPS_NAMES %s

// SAVE_TEMPS_NAMES-NOT: "GNU::Linker"{{.*}}["[[SAVE_TEMPS_INPUT1:.*\.o]]", "[[SAVE_TEMPS_INPUT1]]"]

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_35 \
// RUN:          -save-temps %s -o openmp-offload-gpu 2>&1 \
// RUN:   | FileCheck -check-prefix=TRIPLE %s

// TRIPLE: "-triple" "nvptx64-nvidia-cuda"
// TRIPLE: "-target-cpu" "sm_35"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          --offload-arch=sm_52 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "sm_52"
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (nvptx64-nvidia-cuda)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "device-openmp (nvptx64-nvidia-cuda)" {8}, object
// CHECK-PHASES: 10: clang-offload-packager, {9}, image
// CHECK-PHASES: 11: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {10}, ir
// CHECK-PHASES: 12: backend, {11}, assembler, (host-openmp)
// CHECK-PHASES: 13: assembler, {12}, object, (host-openmp)
// CHECK-PHASES: 14: clang-linker-wrapper, {13}, image, (host-openmp)

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ]]"], output: "[[BINARY:.+.out]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib -save-temps %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEMP-BINDINGS
// CHECK-TEMP-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ:.+]]"], output: "[[BINARY:.+.out]]"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52 --offload-arch=sm_70 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52,sm_70 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52,sm_70,sm_35,sm_80 --no-offload-arch=sm_35,sm_80 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_52]]"], output: "[[DEVICE_OBJ_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_70]]"], output: "[[DEVICE_OBJ_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ_SM_52]]", "[[DEVICE_OBJ_SM_70]]"], output: "[[BINARY:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NVIDIA-AMDGPU

// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[NVIDIA_PTX:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[NVIDIA_PTX]]"], output: "[[NVIDIA_CUBIN:.+]]"
// CHECK-NVIDIA-AMDGPU: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[AMD_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[NVIDIA_CUBIN]]", "[[AMD_BC]]"], output: "[[BINARY:.*]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -x ir -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp --offload-arch=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT_IR:.+]]"], output: "[[OBJECT:.+]]"
// CHECK-IR: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OBJECT]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: "-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"{{.*}}"-emit-llvm"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvida-cuda -march=sm_70 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-new-nvptx-test.bc \
// RUN:          -nogpulib %s -o openmp-offload-gpu 2>&1 \
// RUN:   | FileCheck -check-prefix=DRIVER_EMBEDDING %s

// DRIVER_EMBEDDING: -fembed-offload-object={{.*}}.out

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-host-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HOST-ONLY
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[OUTPUT:.*]]"
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-device-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_ASM:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_ASM]]"], output: "{{.*}}-openmp-nvptx64-nvidia-cuda.o"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-device-only -E -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY-PP
// CHECK-DEVICE-ONLY-PP: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.*]]"], output: "-"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-LIBRARY %s

// CHECK-LTO-LIBRARY: {{.*}}-lomptarget{{.*}}-lomptarget.devicertl

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-NO-LIBRARY %s

// CHECK-NO-LIBRARY-NOT: {{.*}}-lomptarget{{.*}}-lomptarget.devicertl

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xoffload-linker a -Xoffload-linker-nvptx64-nvidia-cuda b -Xoffload-linker-nvptx64 c \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-XLINKER %s

// CHECK-XLINKER: -device-linker=a{{.*}}-device-linker=nvptx64-nvidia-cuda=b{{.*}}-device-linker=nvptx64-nvidia-cuda=c{{.*}}--

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-FEATURES %s

// CHECK-LTO-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx{{[0-9]+}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda --cuda-feature=+ptx64 -foffload-lto %s 2>&1 \
// RUN:    | FileCheck --check-prefix=CHECK-SET-FEATURES %s

// CHECK-SET-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx64

//
// Check that `-Xarch_host` works for OpenMP offloading.
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda -Xarch_host -O3 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=XARCH-HOST %s
// XARCH-HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-O3"
// XARCH-HOST-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-O3"

//
// Check that `-Xarch_device` works for OpenMP offloading.
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda -Xarch_device -O3 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=XARCH-DEVICE %s
// XARCH-DEVICE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-O3"
// XARCH-DEVICE-NOT: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-O3"
