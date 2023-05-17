// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS
// CHECK-HEADERS: "-cc1"{{.*}}"-internal-isystem" "{{.*}}openmp_wrappers" "-include" "__clang_openmp_device_functions.h"
// CHECK-HEADERS: "-cc1"{{.*}}"-internal-isystem" "{{.*}}openmp_wrappers" "-include" "__clang_openmp_device_functions.h"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -nobuiltininc \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-BUILTIN
// CHECK-HEADERS-BUILTIN: "-cc1"{{.*}}"-include" "__clang_openmp_device_functions.h"
// CHECK-HEADERS-BUILTIN: "-cc1"{{.*}}"-include" "__clang_openmp_device_functions.h"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -nostdinc \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -nogpuinc \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// CHECK-HEADERS-DISABLED-NOT: "-cc1"{{.*}}"__clang_openmp_device_functions.h"
