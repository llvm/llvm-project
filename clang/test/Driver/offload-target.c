// RUN: %clang -### -fsycl --offload-targets=spirv64 -nogpuinc %s -ccc-print-bindings 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL
// SYCL: "spirv64" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[SYCL_BC:.+]]"

// RUN: %clang -### --offload-targets=amdgcn-amd-amdhsa -nogpulib -nogpuinc -x hip %s -ccc-print-bindings 2>&1 \
// RUN: | FileCheck %s -check-prefix=HIP
// HIP: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[AMD_OBJ:.+]]"

// RUN: %clang -### --offload-targets=nvptx64-nvidia-cuda -nogpulib -nogpuinc -x cuda %s -ccc-print-bindings 2>&1 \
// RUN: | FileCheck %s -check-prefix=CUDA
// CUDA: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[NV_OBJ:.+]]"

// RUN: %clang -### --offload-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda -fopenmp \
// RUN:   -Xarch_amdgcn --offload-arch=gfx90a -Xarch_nvptx64 --offload-arch=sm_89 \
// RUN:   -nogpulib -nogpuinc %s -ccc-print-bindings 2>&1 \
// RUN: | FileCheck %s -check-prefix=OPENMP
// OPENMP: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[AMD_OBJ:.+]]"
// OPENMP: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]"], output: "[[NV_OBJ:.+]]"

// RUN: %clang -### --offload-targets=spirv64-amd-amdhsa -nogpulib -nogpuinc -x hip %s -ccc-print-bindings 2>&1 \
// RUN: | FileCheck %s -check-prefix=HIPSPIRV
// HIPSPIRV: "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[AMD_OBJ:.+]]"
