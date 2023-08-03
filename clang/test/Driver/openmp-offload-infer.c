// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:          -nogpulib --offload-arch=sm_52 --offload-arch=gfx803 \
// RUN:          --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "gfx803"
// CHECK: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "sm_52"
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp \
// RUN:     --offload-new-driver --offload-arch=sm_70 --offload-arch=gfx908:sramecc+:xnack- \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NVIDIA-AMDGPU

// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[AMD_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[NVIDIA_PTX:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[NVIDIA_PTX]]"], output: "[[NVIDIA_CUBIN:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[AMD_BC]]", "[[NVIDIA_CUBIN]]"], output: "[[BINARY:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp --offload-new-driver \
// RUN:     --offload-arch=sm_52 --offload-arch=sm_70 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS

// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_52]]"], output: "[[DEVICE_OBJ_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_70]]"], output: "[[DEVICE_OBJ_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ_SM_52]]", "[[DEVICE_OBJ_SM_70]]"], output: "[[BINARY:.+]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp \
// RUN:     --offload-arch=sm_70 --offload-arch=gfx908 --offload-arch=skylake \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-FAILED

// CHECK-FAILED: error: failed to deduce triple for target architecture 'skylake'; specify the triple using '-fopenmp-targets' and '-Xopenmp-target' instead.

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp \
// RUN:     --offload-arch=sm_70 --offload-arch=gfx908 -fno-openmp \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED

// CHECK-DISABLED-NOT: "nvptx64-nvidia-cuda" - "clang",
