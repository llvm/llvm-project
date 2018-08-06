///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

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

/// Check cubin file generation and usage by nvlink
// RUN:   %clang -### -no-canonical-prefixes -target powerpc64le-unknown-linux-gnu -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda -save-temps %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-NVLINK %s
/// Check cubin file generation and usage by nvlink when toolchain has BindArchAction
// RUN:   %clang -### -no-canonical-prefixes -target x86_64-apple-darwin17.0.0 -fopenmp=libomp \
// RUN:          -fopenmp-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-NVLINK %s

// CHK-CUBIN-NVLINK: clang{{.*}}" "-o" "[[PTX:.*\.s]]"
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
// RUN:          -no-canonical-prefixes -save-temps %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CUBIN-UNBUNDLING-NVLINK %s

/// Use DAG to ensure that cubin file has been unbundled.
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: nvlink{{.*}}" {{.*}}"[[CUBIN:.*\.cubin]]"
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG: clang-offload-bundler{{.*}}" "-type=o" {{.*}}"-outputs={{.*}}[[CUBIN]]
// CHK-CUBIN-UNBUNDLING-NVLINK-DAG-SAME: "-unbundle"

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

// CHK-BCLIB: clang{{.*}}-triple{{.*}}nvptx64-nvidia-cuda{{.*}}-mlink-cuda-bitcode{{.*}}libomptarget-nvptx-sm_20.bc
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
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O0 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g -O3 --no-cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -g0 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb0 -O3 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -gline-tables-only 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG -check-prefix=LINE_TABLE %s
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 %s -ggdb1 -O2 --cuda-noopt-device-debug 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_DEBUG -check-prefix=LINE_TABLE %s

// NO_DEBUG: ptxas
// LINE_TABLE: "-lineinfo"
// NO_DEBUG-NOT: "-g"
// NO_DEBUG: nvlink
// NO_DEBUG-NOT: "-g"

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

// HAS_DEBUG: "-triple" "nvptx64-nvidia-cuda"
// HAS_DEBUG-SAME: "-dwarf-version=2"
// HAS_DEBUG-SAME: "-fopenmp-is-device"
// HAS_DEBUG: ptxas
// HAS_DEBUG-SAME: "-g"
// HAS_DEBUG-SAME: "--dont-merge-basicblocks"
// HAS_DEBUG-SAME: "--return-at-end"
// HAS_DEBUG: nvlink
// HAS_DEBUG-SAME: "-g"

