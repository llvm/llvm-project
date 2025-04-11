// Check that if we are compiling with fgpu-rdc amdgpu-enable-hipstdpar is not
// passed to CC1, to avoid eager, per TU, removal of potentially accessible
// functions.

// RUN: %clang -### --hipstdpar --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   2>&1 | FileCheck -check-prefix=NORDC %s
// NORDC: {{".*clang.*".* "-triple" "amdgcn-amd-amdhsa".* "-mllvm" "-amdgpu-enable-hipstdpar".*}}

// RUN: %clang -### --hipstdpar --offload-arch=gfx906 %s -nogpulib -nogpuinc -fgpu-rdc \
// RUN:   2>&1 | FileCheck -check-prefix=RDC %s
// RDC-NOT: {{"-mllvm" "-amdgpu-enable-hipstdpar".*}}
