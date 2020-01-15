///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

/// ###########################################################################

/// PTXAS is passed -c flag by default when offloading to an NVIDIA device using OpenMP
/// Check that the flag is passed when -fopenmp-relocatable-target is used.
// RUN:   %clangxx -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -save-temps -no-canonical-prefixes %s -x c++ -c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RTTI %s

// CHK-RTTI: clang{{.*}}" "-triple" "nvptx64-nvidia-cuda"
// CHK-RTTI-SAME: "-fno-rtti"

