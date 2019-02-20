// REQUIRES: clang-driver
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

//
// Verify that CUDA device commands do not get OpenMP flags.
//
// RUN: %clang -no-canonical-prefixes -### -x cuda -target powerpc64le-linux-gnu -std=c++11 --cuda-gpu-arch=sm_35 -fopenmp=libomp %s 2>&1 \
// RUN: | FileCheck %s --check-prefix NO-OPENMP-FLAGS-FOR-CUDA-DEVICE
//
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE:      clang{{.*}}" "-cc1" "-triple" "nvptx64-nvidia-cuda"
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE-NOT:  -fopenmp
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE-NEXT: ptxas" "-m64"
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE-NEXT: fatbinary" "--cuda" "-64"
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE-NEXT: clang{{.*}}" "-cc1" "-triple" "powerpc64le-unknown-linux-gnu"
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE:      -fopenmp
// NO-OPENMP-FLAGS-FOR-CUDA-DEVICE-NEXT: {{ld(.exe)?"}} {{.*}}"-m" "elf64lppc"
