// Tests for SPIRVOpenMPToolChain: driver-level compilation and linking phases
// for OpenMP offloading to AMDGCN-flavored SPIR-V targets.
//
// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target

//===----------------------------------------------------------------------===//
// 1. spirv64 target is routed to SPIRVOpenMPToolChain
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-SPIRV64

// CHECK-SPIRV64:      "-cc1" "-triple" "spirv64"
// CHECK-SPIRV64-SAME: "-fopenmp"
// CHECK-SPIRV64:      "-cc1" "-triple" "x86_64
// CHECK-SPIRV64-SAME: "-fopenmp"

//===----------------------------------------------------------------------===//
// 2. spirv64-amd-amdhsa routes through the AMDHSA isSPIRV() branch in Driver
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64-amd-amdhsa -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-SPIRV64-AMD

// CHECK-SPIRV64-AMD:      "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-SPIRV64-AMD-SAME: "-fopenmp"

//===----------------------------------------------------------------------===//
// 3. --offload-arch=amdgcnspirv expands to spirv64-amd-amdhsa
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   --offload-arch=amdgcnspirv -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-AMDGCNSPIRV

// CHECK-AMDGCNSPIRV:      "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-AMDGCNSPIRV-SAME: "-fopenmp"

//===----------------------------------------------------------------------===//
// 4. Host triple is propagated as -aux-triple for device compilation
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64-amd-amdhsa -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-AUXTRIPLE

// CHECK-AUXTRIPLE:      "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-AUXTRIPLE-SAME: "-aux-triple" "x86_64{{.*}}linux-gnu"

//===----------------------------------------------------------------------===//
// 5. Loop and SLP vectorisation are disabled for device code
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NOVECTORIZE

// CHECK-NOVECTORIZE:      "-cc1" "-triple" "spirv64"
// CHECK-NOVECTORIZE-SAME: "-mllvm" "-vectorize-loops=false"
// CHECK-NOVECTORIZE-SAME: "-mllvm" "-vectorize-slp=false"

//===----------------------------------------------------------------------===//
// 6. Hidden visibility is the default for device symbols
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-VISIBILITY

// CHECK-VISIBILITY:      "-cc1" "-triple" "spirv64"
// CHECK-VISIBILITY-SAME: "-fvisibility=hidden"
// CHECK-VISIBILITY-SAME: "-fapply-global-visibility-to-externs"

//===----------------------------------------------------------------------===//
// 7. An explicit -fvisibility= suppresses the hidden-visibility default
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -fvisibility=default -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-VISIBILITY-OVERRIDE

// CHECK-VISIBILITY-OVERRIDE:      "-cc1" "-triple" "spirv64"
// CHECK-VISIBILITY-OVERRIDE-NOT:  "-fvisibility=hidden"

//===----------------------------------------------------------------------===//
// 8. Linker pipeline: llvm-link then amd-llvm-spirv (or llvm-spirv fallback)
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-LINK

// CHECK-LINK:      llvm-link
// CHECK-LINK:      {{amd-llvm-spirv|llvm-spirv}}
// CHECK-LINK-SAME: "--spirv-max-version=1.6"
// CHECK-LINK-SAME: "--spirv-ext=+all"
// CHECK-LINK-SAME: "--spirv-allow-unknown-intrinsics"
// CHECK-LINK-SAME: "--spirv-lower-const-expr"
// CHECK-LINK-SAME: "--spirv-preserve-auxdata"
// CHECK-LINK-SAME: "--spirv-debug-info-version=nonsemantic-shader-200"

//===----------------------------------------------------------------------===//
// 9. Device library is discovered via the sysroot lib directory
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 \
// RUN:   --sysroot=%S/Inputs/spirv-openmp %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-DEVLIB

// CHECK-DEVLIB: "-mlink-builtin-bitcode" "{{.*}}libomptarget-spirv.bc"

//===----------------------------------------------------------------------===//
// 10. --libomptarget-spirv-bc-path overrides the default search path
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 \
// RUN:   --libomptarget-spirv-bc-path=%S/Inputs/spirv-openmp/lib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-BCPATH

// CHECK-BCPATH: "-mlink-builtin-bitcode" "{{.*}}libomptarget-spirv.bc"

//===----------------------------------------------------------------------===//
// 11. -nogpulib (--no-offloadlib) suppresses device library lookup entirely
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NOGPULIB

// CHECK-NOGPULIB-NOT: libomptarget-spirv.bc

//===----------------------------------------------------------------------===//
// 12. --rocm-path is forwarded to the device toolchain lookup
//===----------------------------------------------------------------------===//

// RUN: %clang -### --target=x86_64-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=spirv64-amd-amdhsa --rocm-path=/opt/rocm \
// RUN:   -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-ROCM

// CHECK-ROCM: "-cc1" "-triple" "spirv64-amd-amdhsa"

//===----------------------------------------------------------------------===//
// Minimal OpenMP target region used by the tests above.
//===----------------------------------------------------------------------===//

int main(void) {
  int a[64];
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 64; ++i)
    a[i] = i;
  return a[0];
}
