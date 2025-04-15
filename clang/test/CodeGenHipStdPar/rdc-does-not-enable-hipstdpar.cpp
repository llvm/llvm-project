// Check that if we are compiling with fgpu-rdc amdgpu-enable-hipstdpar is not
// passed to CC1, to avoid eager, per TU, removal of potentially accessible
// functions.

// RUN: %clang -### --hipstdpar --offload-arch=gfx906 -nogpulib -nogpuinc %s \
// RUN:    --hipstdpar-path=%S/../Driver/Inputs/hipstdpar \
// RUN:    --hipstdpar-thrust-path=%S/../Driver/Inputs/hipstdpar/thrust \
// RUN:    --hipstdpar-prim-path=%S/../Driver/Inputs/hipstdpar/rocprim 2>&1 \
// RUN:    | FileCheck %s -check-prefix=NORDC
// NORDC: {{.*}}"-mllvm" "-amdgpu-enable-hipstdpar"

// RUN: %clang -### --hipstdpar --offload-arch=gfx906 -nogpulib -nogpuinc %s \
// RUN:    -fgpu-rdc --hipstdpar-path=%S/../Driver/Inputs/hipstdpar \
// RUN:    --hipstdpar-thrust-path=%S/../Driver/Inputs/hipstdpar/thrust \
// RUN:    --hipstdpar-prim-path=%S/../Driver/Inputs/hipstdpar/rocprim 2>&1 \
// RUN:    | FileCheck %s -check-prefix=RDC
// RDC-NOT: {{.*}}"-mllvm" "-amdgpu-enable-hipstdpar"
