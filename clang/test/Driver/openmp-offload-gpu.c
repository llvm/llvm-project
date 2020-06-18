///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

/// ###########################################################################

/// Check -Xopenmp-target uses one of the archs provided when several archs are used.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target -march=sm_35 -Xopenmp-target -march=sm_60 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-ARCHS %s

// CHK-FOPENMP-TARGET-ARCHS: ptxas{{.*}}" "--gpu-name" "sm_60"
// CHK-FOPENMP-TARGET-ARCHS: nvlink{{.*}}" "-arch" "sm_60"

/// ###########################################################################

/// Check -Xopenmp-target -march=sm_35 works as expected when two triples are present.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp \
// RUN:          -fopenmp-targets=powerpc64le-ibm-linux-gnu,nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_35 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-COMPILATION %s

// CHK-FOPENMP-TARGET-COMPILATION: ptxas{{.*}}" "--gpu-name" "sm_35"
// CHK-FOPENMP-TARGET-COMPILATION: nvlink{{.*}}" "-arch" "sm_35"

/// ###########################################################################

/// Check that -lomptarget-nvptx is passed to nvlink.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NVLINK %s
/// Check that the value of --libomptarget-nvptx-path is forwarded to nvlink.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp \
// RUN:          --libomptarget-nvptx-path=/path/to/libomptarget/ \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-NVLINK,CHK-LIBOMPTARGET-NVPTX-PATH %s

// CHK-NVLINK: nvlink
// CHK-LIBOMPTARGET-NVPTX-PATH-SAME: "-L/path/to/libomptarget/"
// CHK-NVLINK-SAME: "-lomptarget-nvptx"

/// ###########################################################################

/// Check cubin file generation and usage by nvlink
// RUN:   %clang -### -no-canonical-prefixes -target powerpc64le-unknown-linux-gnu -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-NVLINK %s
/// Check cubin file generation and usage by nvlink when toolchain has BindArchAction
// RUN:   %clang -### -no-canonical-prefixes -target x86_64-apple-darwin17.0.0 -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-NVLINK %s

// CHK-CUBIN-NVLINK: clang{{.*}}" {{.*}}"-fopenmp-is-device" {{.*}}"-o" "[[PTX:.*\.s]]"
// CHK-CUBIN-NVLINK-NEXT: ptxas{{.*}}" "--output-file" "[[CUBIN:.*\.cubin]]" {{.*}}"[[PTX]]"
// CHK-CUBIN-NVLINK-NEXT: nvlink{{.*}}" {{.*}}"[[CUBIN]]"

/// ###########################################################################

/// Check unbundlink of assembly file, cubin file generation and usage by nvlink
// RUN:   touch %t.s
// RUN:   %clang -### -target powerpc64le-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -no-canonical-prefixes -save-temps %t.s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UNBUNDLING-PTXAS-CUBIN-NVLINK %s

/// Use DAG to ensure that assembly file has been unbundled.
// CHK-UNBUNDLING-PTXAS-CUBIN-NVLINK-DAG: ptxas{{.*}}" "--output-file" "[[CUBIN:.*\.cubin]]" {{.*}}"[[PTX:.*\.s]]"
// CHK-UNBUNDLING-PTXAS-CUBIN-NVLINK-DAG: clang-offload-bundler{{.*}}" "-type=s" {{.*}}"-outputs={{.*}}[[PTX]]
// CHK-UNBUNDLING-PTXAS-CUBIN-NVLINK-DAG-SAME: "-unbundle"
// CHK-UNBUNDLING-PTXAS-CUBIN-NVLINK: nvlink{{.*}}" {{.*}}"[[CUBIN]]"

/// ###########################################################################

/// Check cubin file generation and bundling
// RUN:   %clang -### -target powerpc64le-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -no-canonical-prefixes -save-temps %s -c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-CUBIN-BUNDLING %s

// CHK-PTXAS-CUBIN-BUNDLING: clang{{.*}}" "-o" "[[PTX:.*\.s]]"
// CHK-PTXAS-CUBIN-BUNDLING-NEXT: ptxas{{.*}}" "--output-file" "[[CUBIN:.*\.cubin]]" {{.*}}"[[PTX]]"
// CHK-PTXAS-CUBIN-BUNDLING: clang-offload-bundler{{.*}}" "-type=o" {{.*}}"-inputs={{.*}}[[CUBIN]]

/// ###########################################################################

/// Check cubin file unbundling and usage by nvlink
// RUN:   touch %t.o
// RUN:   %clang -### -target powerpc64le-unknown-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -no-canonical-prefixes -save-temps %t.o %S/Inputs/in.so 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-UNBUNDLING-NVLINK %s

/// Use DAG to ensure that cubin file has been unbundled.
// CHK-CUBIN-UNBUNDLING-NVLINK-NOT: clang-offload-bundler{{.*}}" "-type=o"{{.*}}in.so
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: nvlink{{.*}}" {{.*}}"[[CUBIN:.*\.cubin]]"
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: clang-offload-bundler{{.*}}" "-type=o" {{.*}}"-outputs={{.*}}[[CUBIN]]
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG-SAME: "-unbundle"
// CHK-CUBIN-UNBUNDLING-NVLINK-NOT: clang-offload-bundler{{.*}}" "-type=o"{{.*}}in.so

/// ###########################################################################

/// Check cubin file generation and usage by nvlink
// RUN:   touch %t1.o
// RUN:   touch %t2.o
// RUN:   %clang -### -no-canonical-prefixes -target powerpc64le-unknown-linux-gnu -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TWOCUBIN %s
/// Check cubin file generation and usage by nvlink when toolchain has BindArchAction
// RUN:   %clang -### -no-canonical-prefixes -target x86_64-apple-darwin17.0.0 -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TWOCUBIN %s

// CHK-TWOCUBIN: nvlink{{.*}}openmp-offload-{{.*}}.cubin" "{{.*}}openmp-offload-{{.*}}.cubin"

/// ###########################################################################

/// Check PTXAS is passed -c flag when offloading to an NVIDIA device using OpenMP.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-DEFAULT %s

// CHK-PTXAS-DEFAULT: ptxas{{.*}}" "-c"

/// ###########################################################################

/// PTXAS is passed -c flag by default when offloading to an NVIDIA device using OpenMP - disable it.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -fnoopenmp-relocatable-target \
// RUN:          -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-NORELO %s

// CHK-PTXAS-NORELO-NOT: ptxas{{.*}}" "-c"

/// ###########################################################################

/// PTXAS is passed -c flag by default when offloading to an NVIDIA device using OpenMP
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-relocatable-target \
// RUN:          -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PTXAS-RELO %s

// CHK-PTXAS-RELO: ptxas{{.*}}" "-c"

/// ###########################################################################

/// Check that error is not thrown by toolchain when no cuda lib flag is used.
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 \
// RUN:   -nocudalib -fopenmp-relocatable-target -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FLAG-NOLIBDEVICE %s

// CHK-FLAG-NOLIBDEVICE-NOT: error:{{.*}}sm_60

/// ###########################################################################

/// Check that error is not thrown by toolchain when no cuda lib device is found when using -S.
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clang -### -S -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 \
// RUN:   -fopenmp-relocatable-target -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NOLIBDEVICE %s

// CHK-NOLIBDEVICE-NOT: error:{{.*}}sm_60

/// ###########################################################################

/// Check that the runtime bitcode library is part of the compile line. Create a bogus
/// bitcode library and add it to the LIBRARY_PATH.
// RUN:   env LIBRARY_PATH=%S/Inputs/libomptarget %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB %s
/// The user can override default detection using --libomptarget-nvptx-path=.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --libomptarget-nvptx-path=%S/Inputs/libomptarget \
// RUN:   -Xopenmp-target -march=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB %s

// CHK-BCLIB: clang{{.*}}-triple{{.*}}nvptx64-nvidia-cuda{{.*}}-mlink-builtin-bitcode{{.*}}libomptarget-nvptx-sm_20.bc
// CHK-BCLIB-NOT: {{error:|warning:}}

/// ###########################################################################

/// Check that the warning is thrown when the libomptarget bitcode library is not found.
/// Libomptarget requires sm_35 or newer so an sm_20 bitcode library should never exist.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -Xopenmp-target -march=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda \
// RUN:   -fopenmp-relocatable-target -save-temps -no-canonical-prefixes %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BCLIB-WARN %s

// CHK-BCLIB-WARN: No library 'libomptarget-nvptx-sm_20.bc' found in the default clang lib directory or in LIBRARY_PATH. Expect degraded performance due to no inlining of runtime functions on target devices.

/// Check that debug info is emitted in dwarf-2
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O1 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG_DIRECTIVES %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g0 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb0 -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -gline-directives-only 2>&1 \
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
// NO_DEBUG-NOT: "-g"
// NO_DEBUG: nvlink
// NO_DEBUG-NOT: "-g"

// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O0 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O0 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g2 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb2 -O0 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g3 -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb3 -O2 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -gline-tables-only 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb1 -O2 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=HAS_DEBUG %s

// HAS_DEBUG-NOT: warning: debug
// HAS_DEBUG: "-triple" "nvptx64-nvidia-cuda"
// HAS_DEBUG-SAME: "-debug-info-kind={{limited|line-tables-only}}"
// HAS_DEBUG-SAME: "-dwarf-version=2"
// HAS_DEBUG-SAME: "-fopenmp-is-device"
// HAS_DEBUG: ptxas
// HAS_DEBUG-SAME: "-g"
// HAS_DEBUG-SAME: "--dont-merge-basicblocks"
// HAS_DEBUG-SAME: "--return-at-end"
// HAS_DEBUG: nvlink
// HAS_DEBUG-SAME: "-g"

// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-mode -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-mode -fopenmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_MODE %s
// CUDA_MODE: clang{{.*}}"-cc1"{{.*}}"-triple" "{{nvptx64-nvidia-cuda|amdgcn-amd-amdhsa}}"
// CUDA_MODE-SAME: "-fopenmp-cuda-mode"
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-mode -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-mode -fno-openmp-cuda-mode 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_CUDA_MODE %s
// NO_CUDA_MODE-NOT: "-{{fno-|f}}openmp-cuda-mode"

// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-force-full-runtime -fopenmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-force-full-runtime -fopenmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=FULL_RUNTIME %s
// FULL_RUNTIME: clang{{.*}}"-cc1"{{.*}}"-triple" "{{nvptx64-nvidia-cuda|amdgcn-amd-amdhsa}}"
// FULL_RUNTIME-SAME: "-fopenmp-cuda-force-full-runtime"
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fno-openmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-force-full-runtime -fno-openmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fno-openmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_FULL_RUNTIME %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx906 %s -fopenmp-cuda-force-full-runtime -fno-openmp-cuda-force-full-runtime 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_FULL_RUNTIME %s
// NO_FULL_RUNTIME-NOT: "-{{fno-|f}}openmp-cuda-force-full-runtime"

// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -fopenmp-cuda-teams-reduction-recs-num=2048 2>&1 \
// RUN:   | FileCheck -check-prefix=CUDA_RED_RECS %s
// CUDA_RED_RECS: clang{{.*}}"-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"
// CUDA_RED_RECS-SAME: "-fopenmp-cuda-teams-reduction-recs-num=2048"

// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OPENMP_NVPTX_WRAPPERS %s
// OPENMP_NVPTX_WRAPPERS: clang{{.*}}"-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"
// OPENMP_NVPTX_WRAPPERS-SAME: "-internal-isystem" "{{.*}}openmp_wrappers"
